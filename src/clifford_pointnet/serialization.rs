//! Weight serialization for CliffordPointNet models
//!
//! Provides JSON save/load for model weights, supporting both
//! SimpleCliffordNet and GPFeatureClassifier.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Serializable weights for SimpleCliffordNet
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimpleCliffordNetWeights {
    pub layer1_w: Vec<Vec<f64>>,
    pub layer1_b: Vec<f64>,
    pub layer2_w: Vec<Vec<f64>>,
    pub layer2_b: Vec<f64>,
    pub classifier_w: Vec<Vec<f64>>,
    pub classifier_b: Vec<f64>,
    pub hidden_dim: usize,
    pub num_classes: usize,
}

/// Serializable weights for GPFeatureClassifier
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GPClassifierWeights {
    pub layer1_w: Vec<Vec<f64>>,
    pub layer1_b: Vec<f64>,
    pub classifier_w: Vec<Vec<f64>>,
    pub classifier_b: Vec<f64>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_classes: usize,
}

impl SimpleCliffordNetWeights {
    pub fn from_model(model: &super::simple_model::SimpleCliffordNet) -> Self {
        SimpleCliffordNetWeights {
            layer1_w: model.layer1_w.clone(),
            layer1_b: model.layer1_b.clone(),
            layer2_w: model.layer2_w.clone(),
            layer2_b: model.layer2_b.clone(),
            classifier_w: model.classifier_w.clone(),
            classifier_b: model.classifier_b.clone(),
            hidden_dim: model.hidden_dim,
            num_classes: model.num_classes,
        }
    }

    pub fn to_model(&self) -> super::simple_model::SimpleCliffordNet {
        super::simple_model::SimpleCliffordNet {
            layer1_w: self.layer1_w.clone(),
            layer1_b: self.layer1_b.clone(),
            layer2_w: self.layer2_w.clone(),
            layer2_b: self.layer2_b.clone(),
            classifier_w: self.classifier_w.clone(),
            classifier_b: self.classifier_b.clone(),
            hidden_dim: self.hidden_dim,
            num_classes: self.num_classes,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))
    }
}

impl GPClassifierWeights {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))
    }
}
