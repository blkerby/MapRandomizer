use actix_web::{web::Bytes, body::MessageBody};
use object_store::{gcp::GoogleCloudStorageBuilder, ObjectStore};
use anyhow::Result;
use log::info;

// Data needed to render the web page for a randomized seed and to use it to patch a ROM.
// The fields `patch_ips` and `seed_html` correspond to the mandatory files "patch.ips" and
// "seed.html". Extra files (e.g. spoiler logs/maps) can also be added which may be referenced
// in "seed.html". We deliberately use a generic, minimally structured format here, to ensure
// compatibility across Map Rando versions. Each of these files is stored in durable object
// storage (currently Google Cloud Storage) to allow players to share seeds.
pub struct Seed {
    pub name: String,
    pub patch_ips: Vec<u8>,
    pub seed_html: Vec<u8>,
    pub extra_files: Vec<SeedExtraFile>,
}

pub struct SeedExtraFile {
    filename: String,
    data: Vec<u8>,
}

pub struct SeedRepository<'a> {
    object_store: Box<dyn ObjectStore>,
    base_path: String,
    zstd_compressor: zstd::bulk::Compressor<'a>,
    zstd_decompressor: zstd::bulk::Decompressor<'a>,
}

impl<'a> SeedRepository<'a> {
    pub fn new(url: &str) -> Result<Self> {
        Ok(Self {
            object_store: Box::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_url(url)
                    .build()?,
            ),
            base_path: "seeds/".to_string(),
            zstd_compressor: zstd::bulk::Compressor::new(15)?,
            zstd_decompressor: zstd::bulk::Decompressor::new()?,
        })
    }

    pub async fn get_file(&mut self, seed_name: &str, filename: &str) -> Result<Bytes> {
        let path = object_store::path::Path::parse(self.base_path.clone() + seed_name + "/" + filename)?;
        let compressed_data = self.object_store.get(&path).await?.bytes().await?;
        let data = self.zstd_decompressor.decompress(&compressed_data, 10_000_000)?;
        Ok(data.try_into_bytes().unwrap())
    }

    pub async fn put_file(&mut self, seed_name: &str, filename: String, data: Vec<u8>) -> Result<()> {
        let path = object_store::path::Path::parse(self.base_path.clone() + seed_name + "/" + filename.as_str())?;
        let compressed_data = self.zstd_compressor.compress(&data)?;
        self.object_store.put(&path, compressed_data.into()).await?;
        Ok(())
    }

    pub async fn put_seed(&mut self, seed: Seed) -> Result<()> {
        info!("Storing seed");
        let mut futures = Vec::new();
        futures.push(self.put_file(&seed.name, "patch.ips".to_string(), seed.patch_ips));
        futures.push(self.put_file(&seed.name, "seed.html".to_string(), seed.seed_html));
        for extra in seed.extra_files {
            futures.push(self.put_file(&seed.name, extra.filename, extra.data));
        }
        let results = futures::future::join_all(futures).await;
        for result in results {
            result?;
        }
        info!("Done storing seed");
        Ok(())
    }
}
