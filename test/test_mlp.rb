# frozen_string_literal: true
require_relative 'test_helper'

class TestMLP < Minitest::Test
  should 'contain 4 layers' do
    a = MLP::Network.new(hidden_layers: [2, 2, 2], output_nodes: 2, inputs: 2)
    assert_equal 4, a.inspect.size
  end

  should 'create an output neuron with 3 weights' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    assert_equal 3, a.inspect.last.last.weights.size
  end

  should 'create a hidden neuron with 3 weights' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    assert_equal 3, a.inspect.first.last.weights.size
  end

  should 'feed forward and set all neurons last outputs' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 2, inputs: 2)
    a.feed_forward([0, 1])
    b = a.inspect.inject([]) do |array, n|
      array << n.last_output
    end
    b.flatten!
    assert !b.include?(nil)
  end

  should 'return an array after feed forward' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 2, inputs: 2)
    assert_kind_of Array, a.feed_forward([0, 1])
  end

  should 'set its neurons deltas' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    a.train([0, 1], [0])
    b = a.inspect.inject([]) do |array, n|
      array << n.delta
    end
    b.flatten!
    assert !b.include?(nil)
  end

  should 'update its output neurons weights' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    before = a.inspect.last.first.weights.inject([]) do |array, n|
      array << n
    end

    a.train([0, 1], [0])
    after = a.inspect.last.first.weights.inject([]) do |array, n|
      array << n
    end
    assert before != after
  end

  should 'update its hidden neurons weights' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    before = a.inspect.first.first.weights.inject([]) do |array, n|
      array << n
    end

    a.train([0, 1], [0])
    after = a.inspect.first.first.weights.inject([]) do |array, n|
      array << n
    end
    assert before != after
  end

  should 'return the error (float) after training' do
    a = MLP::Network.new(hidden_layers: [2], output_nodes: 1, inputs: 2)
    error = a.train([0, 1], [0])
    assert_kind_of Float, error
  end
end
