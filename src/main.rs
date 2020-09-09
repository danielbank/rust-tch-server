#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;

use std::sync::{Mutex};
use tch::nn::{Module};
use tch::{nn, Device, Tensor};
use rocket::request::Form;
use rocket::State;

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

#[derive(Debug, FromForm)]
struct BMI {
  bmi: f32,
}

#[post("/predict", data="<bmi>")]
fn predict(bmi: Form<BMI>, model_mutex: State<Mutex<Box<dyn Module>>>) -> std::string::String {
  let model = model_mutex.lock().unwrap();
  let prediction = model.forward(&Tensor::of_slice(&[bmi.bmi]));
  format!("{}", f32::from(prediction))
}

#[get("/")]
fn index() -> &'static str {
    "Life Expectancy Prediction Server: curl -d 'bmi=<bmi>' -X POST http://localhost:8080/predict"
}

fn main() {
  let mut vs = nn::VarStore::new(Device::Cpu);
  let linear = net(&vs.root());

  let args: Vec<String> = std::env::args().collect();
  assert!(args.len() == 2, "\n
  Linear Regression server program requires a weights file\n
  Usage:\n
  cargo run <existing_weights_file>\n
  ");
  vs.load(args[1].as_str()).unwrap();

  rocket::ignite()
    .manage(Mutex::new(Box::new(linear) as Box<dyn Module>))
    .mount("/", routes![index, predict]).launch();
}
