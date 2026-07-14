use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::recognize::ModelType;

const EMBEDDING_DIM: usize = 512;
const BYTES_PER_EMBEDDING: usize = EMBEDDING_DIM * 4;

pub struct EnrollmentStore {
    store_path: PathBuf,
    max_embeddings: usize,
}

impl ModelType {
    pub fn suffix(&self) -> &'static str {
        match self {
            ModelType::ArcFace => "_arcface",
            ModelType::AdaFace => "_adaface",
        }
    }

    pub fn from_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "_arcface" => Some(ModelType::ArcFace),
            "_adaface" => Some(ModelType::AdaFace),
            _ => None,
        }
    }
}

impl EnrollmentStore {
    pub fn new(store_path: &str, max_embeddings: usize) -> Self {
        Self {
            store_path: PathBuf::from(store_path),
            max_embeddings,
        }
    }

    pub fn ensure_store_dir(&self) -> Result<()> {
        if !self.store_path.exists() {
            fs::create_dir_all(&self.store_path)
                .with_context(|| format!("Failed to create store dir: {:?}", self.store_path))?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let perms = fs::Permissions::from_mode(0o770);
                fs::set_permissions(&self.store_path, perms)?;

                let _ = std::process::Command::new("chown")
                    .args(["root:faceunlock", self.store_path.to_str().unwrap()])
                    .output();
            }
        }
        Ok(())
    }

    pub fn load_embeddings(&self, username: &str, model_type: ModelType) -> Result<Vec<Vec<f32>>> {
        let path = self.user_path(username, model_type);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut file = fs::File::open(&path)
            .with_context(|| format!("Failed to open embedding file: {:?}", path))?;

        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)
            .context("Failed to read embedding count")?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        if count > self.max_embeddings {
            anyhow::bail!(
                "Invalid embedding count {} in file {:?} (max {})",
                count,
                path,
                self.max_embeddings
            );
        }

        let mut embeddings = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = vec![0u8; BYTES_PER_EMBEDDING];
            file.read_exact(&mut buf)
                .context("Failed to read embedding data")?;

            let embedding: Vec<f32> = buf
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    pub fn save_embedding(
        &self,
        username: &str,
        embedding: &[f32],
        model_type: ModelType,
    ) -> Result<()> {
        self.ensure_store_dir()?;

        if embedding.len() != EMBEDDING_DIM {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                EMBEDDING_DIM,
                embedding.len()
            );
        }

        let path = self.user_path(username, model_type);
        let mut embeddings = self.load_embeddings(username, model_type)?;

        if embeddings.len() >= self.max_embeddings {
            embeddings.remove(0);
        }

        embeddings.push(embedding.to_vec());

        let mut file = fs::File::create(&path)
            .with_context(|| format!("Failed to create embedding file: {:?}", path))?;

        let count = embeddings.len() as u32;
        file.write_all(&count.to_le_bytes())?;

        for emb in &embeddings {
            for &val in emb {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        file.flush()?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o640);
            fs::set_permissions(&path, perms)?;

            let _ = std::process::Command::new("chown")
                .args(["root:faceunlock", path.to_str().unwrap()])
                .output();
        }

        Ok(())
    }

    pub fn clear_embeddings(&self, username: &str) -> Result<()> {
        for model_type in &[ModelType::ArcFace, ModelType::AdaFace] {
            let path = self.user_path(username, *model_type);
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("Failed to delete embedding file: {:?}", path))?;
            }
        }
        Ok(())
    }

    pub fn list_users(&self) -> Result<Vec<(String, Vec<(ModelType, usize)>)>> {
        let mut users: HashMap<String, Vec<(ModelType, usize)>> = HashMap::new();

        if !self.store_path.exists() {
            return Ok(Vec::new());
        }

        for entry in fs::read_dir(&self.store_path)
            .with_context(|| format!("Failed to read store dir: {:?}", self.store_path))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Some((username, model_type)) = parse_embedding_filename(&stem) {
                if let Ok(embeddings) = self.load_embeddings(&username, model_type) {
                    users
                        .entry(username)
                        .or_default()
                        .push((model_type, embeddings.len()));
                }
            }
        }

        let mut result: Vec<_> = users.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }

    fn user_path(&self, username: &str, model_type: ModelType) -> PathBuf {
        self.store_path
            .join(format!("{}{}.bin", username, model_type.suffix()))
    }
}

fn parse_embedding_filename(stem: &str) -> Option<(String, ModelType)> {
    for suffix in &["_arcface", "_adaface"] {
        if let Some(username) = stem.strip_suffix(suffix) {
            if !username.is_empty() {
                return Some((username.to_string(), ModelType::from_suffix(suffix)?));
            }
        }
    }

    if !stem.is_empty() {
        return Some((stem.to_string(), ModelType::ArcFace));
    }

    None
}
