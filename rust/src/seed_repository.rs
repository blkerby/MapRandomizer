use actix_web::web::Bytes;
use object_store::{gcp::GoogleCloudStorageBuilder, ObjectStore};
use anyhow::Result;

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

pub struct SeedRepository {
    object_store: Box<dyn ObjectStore>,
}

impl SeedRepository {
    pub fn new() -> Result<Self> {
        Ok(Self {
            object_store: Box::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_url("gs://super-metroid-map-rando/seeds")
                    .build()?,
            ),
        })
    }

    pub async fn get_file(&self, seed_name: &str, filename: &str) -> Result<Bytes> {
        let path = object_store::path::Path::parse(seed_name.to_string() + "/" + filename)?;
        let data = self.object_store.get(&path).await?.bytes().await?;
        Ok(data)
    }

    pub async fn put_file(&self, seed_name: &str, filename: &str, data: Vec<u8>) -> Result<()> {
        let path = object_store::path::Path::parse(seed_name.to_string() + "/" + filename)?;
        self.object_store.put(&path, data.into()).await?;
        Ok(())
    }

    pub async fn put_seed(&self, seed: Seed) -> Result<()> {
        self.put_file(&seed.name, "patch.ips", seed.patch_ips).await?;
        self.put_file(&seed.name, "seed.html", seed.seed_html).await?;
        for extra in seed.extra_files {
            self.put_file(&seed.name, &extra.filename, extra.data).await?;
        }
        Ok(())
    }
}
