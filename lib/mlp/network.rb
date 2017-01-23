# frozen_string_literal: true

module MLP
  class Network

    UPDATE_WEIGHT_VALUE = 0.25

    def initialize(options = {})
      @input_size = options[:inputs]
      @hidden_layers = options[:hidden_layers]
      @number_of_output_nodes = options[:output_nodes]
      setup_network
    end

    def feed_forward(input)
      @network.each_with_index do |layer, layer_index|
        layer.each do |neuron|
          if layer_index == 0
            neuron.fire(input)
          else
            input = @network[layer_index - 1].map(&:last_output)
            neuron.fire(input)
          end
        end
      end
      @network.last.map(&:last_output)
    end

    def train(input, targets)
      # To go back we must go forward
      feed_forward(input)
      compute_deltas(targets)
      update_weights(input)
      calculate_error(targets)
    end

    def inspect
      @network
    end

    private

    def update_weights(input)
      reversed_network = @network.reverse
      reversed_network.each_with_index do |layer, layer_index|
        if layer_index == 0
          update_output_weights(layer, layer_index, input)
        else
          update_hidden_weights(layer, layer_index, input)
        end
      end
    end

    def update_output_weights(layer, _layer_index, input)
      inputs = @hidden_layers.empty? ? input : @network[-2].map(&:last_output)
      layer.each do |neuron|
        neuron.update_weight(inputs, UPDATE_WEIGHT_VALUE)
      end
    end

    def update_hidden_weights(layer, layer_index, original_input)
      inputs = if layer_index == (@network.size - 1)
                 original_input
               else
                 @network.reverse[layer_index + 1].map(&:last_output)
               end
      layer.each do |neuron|
        neuron.update_weight(inputs, UPDATE_WEIGHT_VALUE)
      end
    end

    def compute_deltas(targets)
      reversed_network = @network.reverse
      reversed_network.each_with_index do |layer, layer_index|
        if layer_index == 0
          compute_output_deltas(layer, targets)
        else
          compute_hidden_deltas(layer, targets)
        end
      end
    end

    def compute_output_deltas(layer, targets)
      layer.each_with_index do |neuron, i|
        output = neuron.last_output
        neuron.delta = output * (1 - output) * (targets[i] - output)
      end
    end

    def compute_hidden_deltas(layer, _targets)
      layer.each_with_index do |neuron, neuron_index|
        error = 0
        @network.last.each do |output_neuron|
          error += output_neuron.delta * output_neuron.weights[neuron_index]
        end
        output = neuron.last_output
        neuron.delta = output * (1 - output) * error
      end
    end

    def calculate_error(targets)
      outputs = @network.last.map(&:last_output)
      sum = 0
      targets.each_with_index do |t, index|
        sum += (t - outputs[index])**2
      end
      0.5 * sum
    end

    def setup_network
      @network = []

      # Hidden Layers
      @hidden_layers.each_with_index do |number_of_neurons, index|
        inputs_count = index == 0 ? @input_size : @hidden_layers[index - 1].size
        @network << Layer.new(
           level: index,
           neurons: number_of_neurons.times.map { Neuron.new(inputs_count) }
        )
      end

      # Output layer
      inputs_count = @hidden_layers.empty? ? @input_size : @hidden_layers.last
      @network << Layer.new(
        level: @hidden_layers.count,
        neurons: @number_of_output_nodes.times.map { Neuron.new(inputs_count) }
      )
    end
  end

  class Layer
    attr_accessor :level, :neurons

    def initialize(params)
      @level = params[:level]
      @neurons = params[:neurons]
    end

    def each(&block)
      neurons.each do |n|
        block.call(n)
      end
    end

    def map(&block)
      neurons.map do |n|
        block.call(n)
      end
    end

    def each_with_index(&block)
      neurons.each_with_index do |n, i|
        block.call(n, i)
      end
    end

    def first
      neurons.first
    end

    def last
      neurons.last
    end
  end
end