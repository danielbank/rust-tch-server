use serde::{Deserialize};
use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Reduction, Tensor};
use failure;
use std::fs::File;

const INPUT_SIZE: i64 = 1;
const OUTPUT_SIZE: i64 = 1;
const ALPHA: f64 = 1e-6;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct BMILifeExpectancy {
  bmi: f32,
  life_expectancy: f32,
}

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

fn train(vs: &mut nn::VarStore, linear: impl Module, features: &Tensor, labels: &Tensor, epochs: u32) -> failure::Fallible<()> {
  let mut opt = nn::Sgd::default().build(&vs, ALPHA).unwrap();
  for _idx in 0..epochs {
    let loss = features.apply(&linear).mse_loss(labels, Reduction::Mean);
    println!("loss: {:#?}", loss);
    opt.backward_step(&loss);
  }
  Ok(())
}

fn main() -> failure::Fallible<()> {
  let file = File::open("data/bmi_and_life_expectancy.csv")?;
  let mut reader = csv::Reader::from_reader(file);
  let mut features: Vec<f32> = Vec::new();
  let mut labels: Vec<f32> = Vec::new();
  for result in reader.deserialize() {
    let row: BMILifeExpectancy = result?;
    features.push(row.bmi);
    labels.push(row.life_expectancy);
  }

  let features = Tensor::of_slice(&features).reshape(&[163, 1]);
  let labels = Tensor::of_slice(&labels);

  let mut vs = nn::VarStore::new(Device::Cpu);
  let linear = net(&vs.root());

  let args: Vec<String> = std::env::args().collect();
  assert!(args.len() > 1 && args.len() < 4, "
  Linear Regression training program requires a number of epochs
    Usage:
      cargo run --example train <num_of_epochs>
      cargo run --example train <num_of_epochs> <existing_weights_file>
  ");
  if args.len() == 3 {
    vs.load(args[2].as_str())?;
    let epochs: u32 = args[1].parse()?;
    train(&mut vs, linear, &features, &labels, epochs);
    vs.save(args[2].as_str())?
  } else {
    let epochs: u32 = args[1].parse()?;
    train(&mut vs, linear, &features, &labels, epochs);
    vs.save("weights.pt");
  }

  Ok(())
}
