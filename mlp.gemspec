# frozen_string_literal: true

lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'mlp/version'

Gem::Specification.new do |s|
  s.name = 'mlp'
  s.version = MLP::VERSION

  s.authors = %w(reddavis beastia)
  s.date = '2017-01-19'
  s.description = <<~TEXT
    Multi-Layer Perceptron Neural Network in Ruby(remake of reddavis' project by BEaStia)"
  TEXT
  s.email = 'gophan1992@gmail.com'
  s.extra_rdoc_files = %w(LICENSE README.rdoc)
  s.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end

  s.homepage = 'http://github.com/BEaStia/mlp'
  s.require_paths = ['lib']
  s.summary = 'Multi-Layer Perceptron Neural Network in Ruby'
  s.required_ruby_version = '~> 2.3'

  s.add_development_dependency 'bundler', '~> 1.13'
  s.add_development_dependency 'rake', '~> 10.0'
  s.add_development_dependency 'rubocop', '~> 0.47'
  s.add_development_dependency 'shoulda', '~> 3.5'
end
