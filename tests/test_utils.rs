//! State-of-the-art test utilities for beautiful, informative output
//!
//! This module provides professional test output with:
//! - Progress bars with spinners
//! - Color-coded results
//! - Detailed metrics (time, error, status)
//! - Clean, structured formatting
//! - Real-time progress updates

use colored::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::time::{Duration, Instant};

/// Test result with detailed metrics
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub duration: Duration,
    pub max_error: f64,
    pub details: Vec<String>,
}

impl TestResult {
    #[allow(dead_code)]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            duration: Duration::from_secs(0),
            max_error: 0.0,
            details: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_error(mut self, max_error: f64) -> Self {
        self.max_error = max_error;
        self
    }

    #[allow(dead_code)]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.details.push(detail.into());
        self
    }

    #[allow(dead_code)]
    pub fn passed(mut self) -> Self {
        self.passed = true;
        self
    }

    #[allow(dead_code)]
    pub fn failed(mut self) -> Self {
        self.passed = false;
        self
    }

    #[allow(dead_code)]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Test suite manager for coordinated output
pub struct TestSuite {
    multi: MultiProgress,
    pub results: Vec<TestResult>,  // Made public for assertions
    start_time: Instant,
}

impl TestSuite {
    pub fn new(name: &str) -> Self {
        println!("\n{}", "═".repeat(80).bright_blue().bold());
        println!(
            "{} {}",
            "◆".bright_cyan().bold(),
            name.bright_white().bold()
        );
        println!("{}\n", "═".repeat(80).bright_blue().bold());

        Self {
            multi: MultiProgress::new(),
            results: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Create a new test with a progress bar
    pub fn test(&self, name: &str, total_steps: u64) -> TestRunner {
        let pb = self.multi.add(ProgressBar::new(total_steps));

        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix} {spinner:.cyan} [{bar:30.cyan/blue}] {pos}/{len} [{elapsed_precise}] {msg}")
                .unwrap()
                .progress_chars("█▓▒░ ")
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        );

        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        let icon = "▸".bright_cyan().bold();
        pb.set_prefix(format!("{} {}", icon, name.bright_white()));
        pb.set_message("initializing...".dimmed().to_string());

        TestRunner {
            pb,
            start_time: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Add a completed test result
    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    /// Print final summary
    pub fn finish(&self) {
        let total_duration = self.start_time.elapsed();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = self.results.len() - passed;

        println!("\n{}", "─".repeat(80).bright_blue());
        println!("{}", "TEST SUMMARY".bright_white().bold());
        println!("{}", "─".repeat(80).bright_blue());

        for result in &self.results {
            let status = if result.passed {
                "✓".bright_green().bold()
            } else {
                "✗".bright_red().bold()
            };

            let error_str = if result.max_error > 0.0 {
                format!("max_error={:.2e}", result.max_error)
            } else {
                "exact".to_string()
            };

            let time_color = if result.duration.as_secs() < 1 {
                "green"
            } else if result.duration.as_secs() < 10 {
                "yellow"
            } else {
                "red"
            };

            let time_str = format!("{:.2}s", result.duration.as_secs_f64());
            let colored_time = match time_color {
                "green" => time_str.bright_green(),
                "yellow" => time_str.yellow(),
                _ => time_str.bright_red(),
            };

            let failed_str = if result.passed {
                String::new()
            } else {
                format!("{}", "FAILED".bright_red().bold())
            };

            println!(
                "  {} {} {} {} {}",
                status,
                result.name.bright_white(),
                format!("[{}]", colored_time).dimmed(),
                format!("[{}]", error_str).bright_cyan(),
                failed_str
            );

            // Print details if any
            for detail in &result.details {
                println!("      {} {}", "→".dimmed(), detail.dimmed());
            }
        }

        println!("\n{}", "─".repeat(80).bright_blue());

        let summary_line = if failed == 0 {
            format!(
                "{} {} passed, {} failed in {:.2}s",
                "✓".bright_green().bold(),
                passed.to_string().bright_green().bold(),
                failed.to_string().bright_green(),
                total_duration.as_secs_f64()
            )
        } else {
            format!(
                "{} {} passed, {} failed in {:.2}s",
                "✗".bright_red().bold(),
                passed.to_string().bright_yellow(),
                failed.to_string().bright_red().bold(),
                total_duration.as_secs_f64()
            )
        };

        println!("{}", summary_line);
        println!("{}\n", "═".repeat(80).bright_blue().bold());
    }
}

/// Individual test runner with progress tracking
pub struct TestRunner {
    pb: ProgressBar,
    start_time: Instant,
    name: String,
}

impl TestRunner {
    /// Increment progress with a message
    pub fn step(&self, msg: &str) {
        self.pb.set_message(msg.to_string());
        self.pb.inc(1);
    }

    /// Update message without incrementing (for long-running operations)
    pub fn update(&self, msg: &str) {
        self.pb.set_message(msg.to_string());
    }

    /// Get elapsed time for this test
    #[allow(dead_code)]
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Set a specific step with message
    #[allow(dead_code)]
    pub fn set_step(&self, pos: u64, msg: &str) {
        self.pb.set_message(msg.to_string());
        self.pb.set_position(pos);
    }

    /// Mark test as complete and return result
    pub fn finish(self, passed: bool, max_error: f64) -> TestResult {
        let duration = self.start_time.elapsed();

        if passed {
            self.pb.finish_with_message(
                format!("{} {:.2e}", "completed".bright_green(), max_error)
            );
        } else {
            self.pb.finish_with_message("failed".bright_red().to_string());
        }

        TestResult {
            name: self.name,
            passed,
            duration,
            max_error,
            details: Vec::new(),
        }
    }

    /// Finish with custom result
    #[allow(dead_code)]
    pub fn finish_with_result(self, result: TestResult) -> TestResult {
        let passed = result.passed;
        let max_error = result.max_error;

        if passed {
            self.pb.finish_with_message(
                format!("{} {:.2e}", "completed".bright_green(), max_error)
            );
        } else {
            self.pb.finish_with_message("failed".bright_red().to_string());
        }

        result.with_duration(self.start_time.elapsed())
    }
}

/// Print a section header
#[allow(dead_code)]
pub fn print_section(title: &str) {
    println!("\n{}", title.bright_cyan().bold().underline());
}

/// Print test configuration
pub fn print_config(params: &[(&str, String)]) {
    println!("\n{}", "Configuration:".bright_white().bold());
    for (key, value) in params {
        println!("  {} {}",
            format!("{}:", key).dimmed(),
            value.bright_yellow()
        );
    }
    println!();
}

/// Format error as colored string
#[allow(dead_code)]
pub fn format_error(error: f64, threshold: f64) -> ColoredString {
    if error < threshold {
        format!("{:.2e}", error).bright_green()
    } else if error < threshold * 10.0 {
        format!("{:.2e}", error).yellow()
    } else {
        format!("{:.2e}", error).bright_red()
    }
}

/// Check if error is acceptable and return colored result
#[allow(dead_code)]
pub fn check_error(error: f64, threshold: f64, context: &str) -> Result<(), String> {
    if error < threshold {
        Ok(())
    } else {
        Err(format!(
            "{}: error {:.2e} exceeds threshold {:.2e}",
            context, error, threshold
        ))
    }
}
