use anyhow::{Context, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

const EMBEDDING_DIM: usize = 512;
const MAX_EMBEDDINGS: usize = 5;
const BYTES_PER_EMBEDDING: usize = EMBEDDING_DIM * 4;

pub struct EnrollmentStore {
    store_path: PathBuf,
}

impl EnrollmentStore {
    pub fn new(store_path: &str) -> Self {
        Self {
            store_path: PathBuf::from(store_path),
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

                // Try to set ownership (requires root)
                let _ = std::process::Command::new("chown")
                    .args(["root:faceunlock", self.store_path.to_str().unwrap()])
                    .output();
            }
        }
        Ok(())
    }

    pub fn load_embeddings(&self, username: &str) -> Result<Vec<Vec<f32>>> {
        let path = self.user_path(username);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut file = fs::File::open(&path)
            .with_context(|| format!("Failed to open embedding file: {:?}", path))?;

        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)
            .context("Failed to read embedding count")?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        if count > MAX_EMBEDDINGS {
            anyhow::bail!(
                "Invalid embedding count {} in file {:?} (max {})",
                count,
                path,
                MAX_EMBEDDINGS
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

    pub fn save_embedding(&self, username: &str, embedding: &[f32]) -> Result<()> {
        self.ensure_store_dir()?;

        if embedding.len() != EMBEDDING_DIM {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                EMBEDDING_DIM,
                embedding.len()
            );
        }

        let path = self.user_path(username);
        let mut embeddings = self.load_embeddings(username)?;

        if embeddings.len() >= MAX_EMBEDDINGS {
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
        let path = self.user_path(username);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("Failed to delete embedding file: {:?}", path))?;
        }
        Ok(())
    }

    pub fn list_users(&self) -> Result<Vec<(String, usize)>> {
        let mut users = Vec::new();

        if !self.store_path.exists() {
            return Ok(users);
        }

        for entry in fs::read_dir(&self.store_path)
            .with_context(|| format!("Failed to read store dir: {:?}", self.store_path))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                let username = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                if let Ok(embeddings) = self.load_embeddings(&username) {
                    users.push((username, embeddings.len()));
                }
            }
        }

        users.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(users)
    }

    fn user_path(&self, username: &str) -> PathBuf {
        self.store_path.join(format!("{}.bin", username))
    }
}
