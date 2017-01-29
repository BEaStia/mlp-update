# encoding: utf-8
# frozen_string_literal: true
require_relative 'test_helper'
require 'minitest/autorun'

class TestMlpHard < Minitest::Test
  def prepare(str)
    str.chars.map(&:ord).map { |x| x.to_s(2).chars }.flatten.map(&:to_i)
  end

  def test_russian_words
    a = MLP::Network.new(hidden_layers: [100], output_nodes: 1, inputs: 55)
    error = 1
    sexes = {f: 0, m: 1}
    5000.times do |i|
      a.train(prepare('белая'), [sexes[:f]])
      a.train(prepare('белый'), [sexes[:m]])

      a.train(prepare('синая'), [sexes[:f]])
      a.train(prepare('синий'), [sexes[:m]])

      a.train(prepare('яркая'), [sexes[:f]])
      a.train(prepare('яркий'), [sexes[:m]])

      a.train(prepare('бурая'), [sexes[:f]])
      a.train(prepare('бурый'), [sexes[:m]])

      a.train(prepare('живая'), [sexes[:f]])
      a.train(prepare('живой'), [sexes[:m]])

      a.train(prepare('левая'), [sexes[:f]])
      a.train(prepare('левый'), [sexes[:m]])

      a.train(prepare('немая'), [sexes[:f]])
      a.train(prepare('немой'), [sexes[:m]])

      a.train(prepare('самая'), [sexes[:f]])
      a.train(prepare('самый'), [sexes[:m]])

      a.train(prepare('куцый'), [sexes[:m]])
      a.train(prepare('куцая'), [sexes[:f]])

      error = a.train(prepare('палый'), [sexes[:m]])
      # puts "Error after iteration #{i}:\t#{error}" if i % 200 == 0
    end

    puts 'Test data'
    words = %w(русый тощий милый).freeze
    words.each do |word|
      detected_value = a.feed_forward(prepare(word)).first
      assert_equal detected_value.round, sexes[:m]
    end
    assert error < 0.1
  end
end
