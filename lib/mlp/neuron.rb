# encoding: utf-8
# frozen_string_literal: true

module MLP
  class Neuron
    attr_reader :last_output, :weights
    attr_accessor :delta, :id

    def initialize(number_of_inputs, id)
      create_weights(number_of_inputs)
      @id = id
    end

    def fire(input)
      @last_output = activation_function(input)
    end

    def update_weight(inputs, training_rate)
      inputs << -1 # Add the bias
      @weights.each_index do |i|
        @weights[i] += training_rate * delta * inputs[i]
      end
    end

    def inspect
      @weights
    end

    private

    def activation_function(input)
      sum = 0
      input.each_with_index do |n, index|
        sum += @weights[index] * n
      end
      sum += @weights.last * -1 # bias node
      sigmoid_function(sum)
    end

    # g(h) = 1 / (1+exp(-B*h(j)))
    def sigmoid_function(x)
      1 / (1 + Math.exp(-1 * x))
    end

    def create_weights(number_of_inputs)
      # Create random weights between 0 & 1
      #  Plus another one for the bias node
      @weights = []
      (number_of_inputs + 1).times do
        @weights << (rand > 0.5 ? -rand : rand)
      end
    end
  end
end
