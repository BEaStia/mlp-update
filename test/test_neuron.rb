# frozen_string_literal: true
require 'test_helper'

class TestNeuron < Minitest::Test
  should 'contain 3 weights (including weight for bias node)' do
    a = MLP::Neuron.new(2)
    assert_equal 3, a.inspect.size
  end

  should 'save its last output' do
    a = MLP::Neuron.new(2)
    a.fire([0, 1])
    assert a.last_output
  end
end
