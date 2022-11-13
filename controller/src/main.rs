use clap::{Arg, ArgAction, Command};

fn main() {
    let matches = Command::new("brion")
        .about("Project control system for easier run")
        .version("0.1.0")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .author("commanderxa")
        // run subcommands
        .subcommand(
            Command::new("run")
                .short_flag('R')
                .long_flag("run")
                .about("Run server and client sides")
                .arg(
                    Arg::new("client")
                        .short('C')
                        .long("client")
                        .help("Run only client side")
                        .conflicts_with("server"),
                )
                .arg(
                    Arg::new("server")
                        .short('S')
                        .long("server")
                        .help("Run only server side")
                        .conflicts_with("client"),
                ),
        )
        // install subcommands
        .subcommand(
            Command::new("install")
                .short_flag('i')
                .long_flag("install")
                .about("Condifures the environment and installs all dependencies")
                .arg(
                    Arg::new("client")
                        .short('C')
                        .long("client")
                        .help("Install dependencies only for client side")
                        .conflicts_with("server"),
                )
                .arg(
                    Arg::new("server")
                        .short('S')
                        .long("server")
                        .help("Install dependencies only for server side")
                        .conflicts_with("client"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("run", run_matches)) => {
            if !run_matches.contains_id("client") && !run_matches.contains_id("server") {
                std::process::Command::new("python")
                    .arg("./src/main.py")
                    .output()
                    .expect("Failed running server");

                std::process::Command::new("npm")
                    .arg("start")
                    .output()
                    .expect("Failed running server");
            }
        }
        Some(("install", install_matches)) => {}
        _ => unreachable!(),
    }
}
