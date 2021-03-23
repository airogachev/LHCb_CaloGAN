#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::jit::script::Module module;
  module = torch::jit::load("../gen_script_cpu.pth");
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({1, 261}));
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  std::cout << "ok\n";
}
