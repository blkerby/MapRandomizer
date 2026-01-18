use std::hash::Hash;

use hashbrown::HashMap;

pub fn sorted_hashmap_iter<T, U>(hashmap: &HashMap<T, U>) -> impl Iterator<Item = (&T, &U)>
where
    T: Ord + Hash,
{
    let mut keys: Vec<&T> = hashmap.keys().collect();
    keys.sort();
    keys.into_iter().map(|k| (k, &hashmap[k]))
}
