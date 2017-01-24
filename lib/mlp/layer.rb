# frozen_string_literal: true

module MLP
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

    def last_output
      neurons.map(&:last_output)
    end

    def each_with_index(&block)
      neurons.each_with_index do |n, i|
        block.call(n, i)
      end
    end

    def inject(val, &block)
      neurons.inject(val) do |value, n|
        block.call(value, n)
      end
    end

    def first
      neurons.first
    end

    def last
      neurons.last
    end

    def initial?
      level.zero?
    end

    def delta
      neurons.map(&:delta)
    end

    def update_weights(inputs, training_rate)
      neurons.each { |n| n.update_weight(inputs, training_rate) }
    end
  end
end