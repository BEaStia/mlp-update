# frozen_string_literal: true

module MLP
  class Network

    attr_accessor :input_size, :hidden_layers, :number_of_output_nodes, :network

    UPDATE_WEIGHT_VALUE = 0.25

    def initialize(options = {})
      @input_size = options[:inputs]
      @hidden_layers = options[:hidden_layers]
      @number_of_output_nodes = options[:output_nodes]
      if options[:network].present?
        @network = options[:network]
      else
        setup_network
      end
    end

    def feed_forward(input)
      @network.each do |layer|
        layer.each do |neuron|
          neuron.fire(layer.initial? ? input : @network[layer.level - 1].last_output)
        end
      end
      @network.last.last_output
    end

    def train(input, targets)
      # To go back we must go forward
      feed_forward(input)
      compute_deltas(targets)
      update_weights(input)
      calculate_error(targets)
    end

    def dump
      JSON.dump(
        network: network,
        input_size: input_size,
        hidden_layers: hidden_layers,
        number_of_output_nodes: number_of_output_nodes
      )
    end

    def self.load(json)
      parsed_data = JSON.parse(json)
      Network.new(parsed_data)
    end

    def update_weights(input)
      @network.reverse.each_with_index do |layer, i|
        public_send("update_#{i.zero? ? :output : :hidden}_weights", layer, i, input)
      end
    end

    def update_output_weights(layer, _layer_index, input)
      inputs = @hidden_layers.empty? ? input : @network[-2].last_output
      layer.update_weights(inputs, UPDATE_WEIGHT_VALUE)
    end

    def update_hidden_weights(layer, layer_index, original_input)
      inputs = if layer_index == (@network.size - 1)
                 original_input
               else
                 @network.reverse[layer_index + 1].last_output
               end
      layer.update_weights(inputs, UPDATE_WEIGHT_VALUE)
    end

    def compute_deltas(targets)
      @network.reverse.each_with_index do |layer, i|
        public_send("compute_#{i.zero? ? :output : :hidden}_deltas", layer, targets)
      end
    end

    def compute_output_deltas(layer, targets)
      layer.each do |neuron|
        output = neuron.last_output
        neuron.delta = output * (1 - output) * (targets[neuron.id] - output)
      end
    end

    def compute_hidden_deltas(layer, _targets)
      layer.each do |neuron|
        error = @network.last.inject(0) do |error, output_neuron|
          error + output_neuron.delta * output_neuron.weights[neuron.id]
        end
        output = neuron.last_output
        neuron.delta = output * (1 - output) * error
      end
    end

    def calculate_error(targets)
      outputs = @network.last.last_output
      0.5 * targets.each_with_index.inject(0) do |sum, (t, index)|
        sum + (t - outputs[index])**2
      end
    end

    def setup_network
      @network = []

      # Hidden Layers
      @hidden_layers.each_with_index do |number_of_neurons, index|
        inputs_count = index == 0 ? @input_size : @hidden_layers[index - 1].size
        @network << MLP::Layer.new(
           level: index,
           neurons: number_of_neurons.times.map { |i| Neuron.new(inputs_count, i) }
        )
      end

      # Output layer
      inputs_count = @hidden_layers.empty? ? @input_size : @hidden_layers.last
      @network << MLP::Layer.new(
        level: @hidden_layers.count,
        neurons: @number_of_output_nodes.times.map { |i| Neuron.new(inputs_count, i) }
      )
    end
  end
end