//! Production Storage Configuration
//!
//! Manages persistent storage for ALEN including:
//! - Database path configuration
//! - Automatic database initialization
//! - Migration support
//! - Backup and recovery

use std::path::{Path, PathBuf};
use std::fs;
use std::env;

/// Storage configuration for production deployment
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Base directory for all storage
    pub base_dir: PathBuf,
    /// Episodic memory database path
    pub episodic_db_path: PathBuf,
    /// Semantic memory database path
    pub semantic_db_path: PathBuf,
    /// Conversation history database path
    pub conversation_db_path: PathBuf,
    /// System configuration file path
    pub config_file_path: PathBuf,
    /// Backup directory
    pub backup_dir: PathBuf,
}

impl StorageConfig {
    /// Create production storage configuration
    pub fn production() -> Result<Self, std::io::Error> {
        let base_dir = Self::get_base_dir()?;
        Self::from_base_dir(base_dir)
    }

    /// Create development storage configuration (in temp directory)
    pub fn development() -> Result<Self, std::io::Error> {
        let base_dir = env::temp_dir().join("alen_dev");
        Self::from_base_dir(base_dir)
    }

    /// Create from custom base directory
    pub fn from_base_dir<P: AsRef<Path>>(base_dir: P) -> Result<Self, std::io::Error> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create directory structure
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(base_dir.join("databases"))?;
        fs::create_dir_all(base_dir.join("backups"))?;
        fs::create_dir_all(base_dir.join("config"))?;

        Ok(Self {
            episodic_db_path: base_dir.join("databases/episodic.db"),
            semantic_db_path: base_dir.join("databases/semantic.db"),
            conversation_db_path: base_dir.join("databases/conversations.db"),
            config_file_path: base_dir.join("config/system.json"),
            backup_dir: base_dir.join("backups"),
            base_dir,
        })
    }

    /// Get base directory from environment or use default
    fn get_base_dir() -> Result<PathBuf, std::io::Error> {
        // Check environment variable first
        if let Ok(dir) = env::var("ALEN_DATA_DIR") {
            return Ok(PathBuf::from(dir));
        }

        // Use platform-specific default
        #[cfg(target_os = "linux")]
        {
            if let Ok(home) = env::var("HOME") {
                return Ok(PathBuf::from(home).join(".local/share/alen"));
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(home) = env::var("HOME") {
                return Ok(PathBuf::from(home).join("Library/Application Support/ALEN"));
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = env::var("APPDATA") {
                return Ok(PathBuf::from(appdata).join("ALEN"));
            }
        }

        // Fallback to current directory
        Ok(PathBuf::from("./alen_data"))
    }

    /// Create backup of all databases
    pub fn create_backup(&self) -> Result<PathBuf, std::io::Error> {
        use chrono::Utc;

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let backup_subdir = self.backup_dir.join(&timestamp);
        fs::create_dir_all(&backup_subdir)?;

        // Copy databases
        if self.episodic_db_path.exists() {
            fs::copy(
                &self.episodic_db_path,
                backup_subdir.join("episodic.db"),
            )?;
        }

        if self.semantic_db_path.exists() {
            fs::copy(
                &self.semantic_db_path,
                backup_subdir.join("semantic.db"),
            )?;
        }

        if self.conversation_db_path.exists() {
            fs::copy(
                &self.conversation_db_path,
                backup_subdir.join("conversations.db"),
            )?;
        }

        if self.config_file_path.exists() {
            fs::copy(
                &self.config_file_path,
                backup_subdir.join("system.json"),
            )?;
        }

        Ok(backup_subdir)
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        StorageStats {
            episodic_db_size: Self::file_size(&self.episodic_db_path),
            semantic_db_size: Self::file_size(&self.semantic_db_path),
            conversation_db_size: Self::file_size(&self.conversation_db_path),
            total_size: Self::dir_size(&self.base_dir),
            backup_count: self.count_backups(),
        }
    }

    fn file_size(path: &Path) -> u64 {
        path.metadata().map(|m| m.len()).unwrap_or(0)
    }

    fn dir_size(path: &Path) -> u64 {
        if !path.exists() {
            return 0;
        }

        let mut size = 0;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        size += metadata.len();
                    } else if metadata.is_dir() {
                        size += Self::dir_size(&entry.path());
                    }
                }
            }
        }
        size
    }

    fn count_backups(&self) -> usize {
        if !self.backup_dir.exists() {
            return 0;
        }

        fs::read_dir(&self.backup_dir)
            .map(|entries| entries.count())
            .unwrap_or(0)
    }

    /// Clean old backups, keeping only the most recent N
    pub fn clean_old_backups(&self, keep: usize) -> Result<usize, std::io::Error> {
        if !self.backup_dir.exists() {
            return Ok(0);
        }

        let mut backups: Vec<_> = fs::read_dir(&self.backup_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false))
            .collect();

        // Sort by modification time (newest first)
        backups.sort_by(|a, b| {
            let time_a = a.metadata().ok().and_then(|m| m.modified().ok());
            let time_b = b.metadata().ok().and_then(|m| m.modified().ok());
            time_b.cmp(&time_a)
        });

        let mut deleted = 0;
        for backup in backups.iter().skip(keep) {
            if fs::remove_dir_all(backup.path()).is_ok() {
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

/// Storage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StorageStats {
    pub episodic_db_size: u64,
    pub semantic_db_size: u64,
    pub conversation_db_size: u64,
    pub total_size: u64,
    pub backup_count: usize,
}

impl StorageStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0)
    }

    pub fn episodic_size_mb(&self) -> f64 {
        self.episodic_db_size as f64 / (1024.0 * 1024.0)
    }

    pub fn semantic_size_mb(&self) -> f64 {
        self.semantic_db_size as f64 / (1024.0 * 1024.0)
    }

    pub fn conversation_size_mb(&self) -> f64 {
        self.conversation_db_size as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_development_config() {
        let config = StorageConfig::development().unwrap();
        assert!(config.base_dir.to_str().unwrap().contains("alen_dev"));
    }

    #[test]
    fn test_directory_creation() {
        let temp = env::temp_dir().join("alen_test");
        let config = StorageConfig::from_base_dir(&temp).unwrap();

        assert!(config.base_dir.exists());
        assert!(config.base_dir.join("databases").exists());
        assert!(config.backup_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn test_storage_stats() {
        let config = StorageConfig::development().unwrap();
        let stats = config.get_stats();

        assert_eq!(stats.episodic_db_size, 0); // No db yet
        assert!(stats.total_size_mb() >= 0.0);
    }
}
