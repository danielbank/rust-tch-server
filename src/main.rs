use tch::nn::{Module};
use tch::{nn, Device, Tensor};
use failure;

const INPUT_SIZE: i64 = 1;
const OUTPUT_SIZE: i64 = 1;

fn net(vs: &nn::Path) -> impl Module {
  nn::seq()
  .add(nn::linear(
    vs / "layer1",
    INPUT_SIZE,
    OUTPUT_SIZE,
    nn::LinearConfig{
      ws_init: nn::Init::Const(0.),
      bs_init: Some(nn::Init::Const(0.)),
      bias: true,
    }
  ))
}

fn main() -> failure::Fallible<()> {
  let mut vs = nn::VarStore::new(Device::Cpu);
  let linear = net(&vs.root());

  let args: Vec<String> = std::env::args().collect();
  assert!(args.len() == 2, "\n
  Linear Regression server program requires a weights file\n
  Usage:\n
  cargo run <existing_weights_file>\n
  ");
  vs.load(args[1].as_str())?;

  let bmi: f32 = 28.45698;
  let prediction = linear.forward(&Tensor::of_slice(&[bmi]));
  println!("Predicted a life expectency of {:#?} for {}", prediction, bmi);

  Ok(())
}