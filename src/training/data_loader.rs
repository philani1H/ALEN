//! Data Loading for Training
//!
//! Loads and processes training data from JSON files.
//! Supports multiple data formats and categories.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A single training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input text/question
    pub input: String,
    /// Expected output/answer
    pub output: String,
    /// Reasoning explanation (optional)
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Category of the example
    #[serde(default)]
    pub category: Option<String>,
    /// Subcategory
    #[serde(default)]
    pub subcategory: Option<String>,
    /// Difficulty level (1-10)
    #[serde(default = "default_difficulty")]
    pub difficulty: u8,
}

fn default_difficulty() -> u8 {
    5
}

/// Category of training data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataCategory {
    Mathematics,
    Physics,
    Chemistry,
    Biology,
    ComputerScience,
    Language,
    Logic,
    GeneralKnowledge,
    Conversation,
    Custom(String),
}

impl Default for DataCategory {
    fn default() -> Self {
        DataCategory::GeneralKnowledge
    }
}

/// Training data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// Version of the data format
    #[serde(default = "default_version")]
    pub version: String,
    /// Description of the dataset
    #[serde(default)]
    pub description: Option<String>,
    /// Categorized examples
    #[serde(default)]
    pub categories: HashMap<String, CategoryData>,
    /// Flat list of examples (alternative format)
    #[serde(default)]
    pub examples: Vec<TrainingExample>,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Data for a single category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryData {
    /// Subcategories with their examples
    #[serde(flatten)]
    pub subcategories: HashMap<String, Vec<SubcategoryExample>>,
}

/// Example within a subcategory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubcategoryExample {
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub reasoning: Option<String>,
}

impl TrainingData {
    /// Load training data from JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let data: TrainingData = serde_json::from_str(&content)?;
        Ok(data)
    }

    /// Get all examples as a flat list
    pub fn all_examples(&self) -> Vec<TrainingExample> {
        let mut all = self.examples.clone();

        // Flatten categorized examples
        for (category, cat_data) in &self.categories {
            for (subcategory, examples) in &cat_data.subcategories {
                for ex in examples {
                    all.push(TrainingExample {
                        input: ex.input.clone(),
                        output: ex.output.clone(),
                        reasoning: ex.reasoning.clone(),
                        category: Some(category.clone()),
                        subcategory: Some(subcategory.clone()),
                        difficulty: 5,
                    });
                }
            }
        }

        all
    }

    /// Get examples by category
    pub fn by_category(&self, category: &str) -> Vec<TrainingExample> {
        self.all_examples()
            .into_iter()
            .filter(|ex| ex.category.as_deref() == Some(category))
            .collect()
    }

    /// Get total number of examples
    pub fn len(&self) -> usize {
        self.all_examples().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Data loader for batch training
pub struct DataLoader {
    /// All training examples
    examples: Vec<TrainingExample>,
    /// Current position in the dataset
    position: usize,
    /// Batch size
    batch_size: usize,
    /// Whether to shuffle data
    shuffle: bool,
    /// Random seed for shuffling
    seed: u64,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            position: 0,
            batch_size,
            shuffle: true,
            seed: 42,
        }
    }

    /// Load from training data
    pub fn from_training_data(data: &TrainingData, batch_size: usize) -> Self {
        Self::new(data.all_examples(), batch_size)
    }

    /// Set shuffle mode
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Shuffle the dataset
    pub fn shuffle_data(&mut self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        self.examples.shuffle(&mut rng);
        self.seed = self.seed.wrapping_add(1);
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<Vec<&TrainingExample>> {
        if self.position >= self.examples.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.examples.len());
        let batch: Vec<&TrainingExample> = self.examples[self.position..end]
            .iter()
            .collect();

        self.position = end;
        Some(batch)
    }

    /// Reset to beginning of dataset
    pub fn reset(&mut self) {
        self.position = 0;
        if self.shuffle {
            self.shuffle_data();
        }
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.examples.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get total examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Iterate over all batches
    pub fn iter_batches(&mut self) -> BatchIterator<'_> {
        self.reset();
        BatchIterator { loader: self }
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    loader: &'a mut DataLoader,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Vec<TrainingExample>;

    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next_batch().map(|batch| {
            batch.into_iter().cloned().collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example() {
        let example = TrainingExample {
            input: "What is 2+2?".to_string(),
            output: "4".to_string(),
            reasoning: Some("Basic addition".to_string()),
            category: Some("math".to_string()),
            subcategory: Some("arithmetic".to_string()),
            difficulty: 1,
        };

        assert_eq!(example.input, "What is 2+2?");
        assert_eq!(example.output, "4");
    }

    #[test]
    fn test_data_loader() {
        let examples = vec![
            TrainingExample {
                input: "Q1".to_string(),
                output: "A1".to_string(),
                reasoning: None,
                category: None,
                subcategory: None,
                difficulty: 5,
            },
            TrainingExample {
                input: "Q2".to_string(),
                output: "A2".to_string(),
                reasoning: None,
                category: None,
                subcategory: None,
                difficulty: 5,
            },
        ];

        let mut loader = DataLoader::new(examples, 1);
        
        let batch1 = loader.next_batch();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 1);

        let batch2 = loader.next_batch();
        assert!(batch2.is_some());

        let batch3 = loader.next_batch();
        assert!(batch3.is_none());
    }
}
