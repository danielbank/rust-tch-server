# rust-tch-server

Example of a Rocket web server running a [tch-rs](https://crates.io/crates/tch) Linear Regression model that predicts life expectancy based on body mass index.  The project consists of two parts: a

- a training program that reads the `bmi_and_life_expectancy.csv` data, trains a Linear Regression model, and saves its weights to file
- a web server that instantiates a Linear Regression model using the saved weights and responds to requests with predictions.

This project was presented at the [Desert Rust](https://rust.azdevs.org/) meetup.

## Installation

See the **Getting Started** section of the [tch crate](https://crates.io/crates/tch) for instructions on how to set up Libtorch on your host, which is necessary to run this example.

## Usage

### Train a Linear Regression Model

- Running the `train` example will generate a `weights.pt` file with the model weights (i.e. the slope and y-intercept for a line):

```
cargo run --example train <num_of_epochs>
```

- If you already have a `weights.pt` file, you can also specify it and the training will pick up where it left off:

```
cargo run --example train <number_of_epochs> weights.pt
```

### Run the Prediction Server

- Once you have a weights file, you can run the server:

```
cargo run weights.pt
```

- BMI can be posted to the `/predict` route and it will respond with a predicted life expectancy:

```
curl -d 'bmi=28.45' -X POST http://localhost:8080/predict
```

## Troubleshooting

### Sharing the model via `rocket::State` but `impl Module` cannot be shared between threads safely

The main challenge I faced was when I wanted to utilize the linear regression model in the prediction route.  The model is created in `main()`:

```
let mut vs = nn::VarStore::new(Device::Cpu);
let linear = net(&vs.root());
```

The first (naive) approach I attempted was to pass it directly to the prediction route by having Rocket manage it as state:

```
rocket::ignite()
  .manage(linear)
  .mount("/", routes![index]).launch();
```

The type of `linear` is `impl Module` or "a type that implement nn::Module".  As of Rust 1.27, there is a [new syntax](https://doc.rust-lang.org/edition-guide/rust-2018/trait-system/dyn-trait-for-trait-objects.html) for using trait objects, `dyn Module`.  I use both interchangeably in the code.  There is probably a reason to prefer one over the other but I do not know it.  It seems like `dyn Module` is the preferred syntax.

In any case, the trouble arises in the definition for the route handler, where we now write:

```
fn predict(bmi: Form<BMI>, model_state: State<impl Module>) -> std::string::String { ... }
```

This line then reports an error:

```
`impl Module` cannot be shared between threads safely

`impl Module` cannot be shared between threads safely

help: consider further restricting this bound: ` + std::marker::Sync`rustc(E0277)
main.rs(33, 46): `impl Module` cannot be shared between threads safely
state.rs(106, 32): required by this bound in `rocket::State`
`impl Trait` not allowed outside of function and inherent method return types
```

In this case, the advice for this error is not particularly helpful because further restricting the bounds only changes the error:

```
the size for values of type `(dyn tch::nn::Module + std::marker::Sync + 'static)` cannot be known at compilation time

doesn't have a size known at compile-time

help: the trait `std::marker::Sized` is not implemented for `(dyn tch::nn::Module + std::marker::Sync + 'static)`
```

The solution is to use a [mutex](https://doc.rust-lang.org/std/sync/struct.Mutex.html) to ensure that shared access to the model in the route and main thread is protected.  The `main()` function becomes:

```
rocket::ignite()
  .manage(Mutex::new(Box::new(linear) as Box<dyn Module>))
  .mount("/", routes![index, predict]).launch();
```

And the definition of the route handler becomes:

```
fn predict(bmi: Form<BMI>, model_mutex: State<Mutex<Box<dyn Module>>>) -> std::string::String { ... }
```