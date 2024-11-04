use hashbrown::HashMap;
use maprando_game::{IndexedVec, Item};

use crate::{randomize::ItemPriorityGroup, settings::KeyItemPriority};

pub fn get_item_priorities(
    item_priorities: &HashMap<Item, KeyItemPriority>,
) -> Vec<ItemPriorityGroup> {
    let mut priorities: IndexedVec<KeyItemPriority> = IndexedVec::default();
    priorities.add(&KeyItemPriority::Early);
    priorities.add(&KeyItemPriority::Default);
    priorities.add(&KeyItemPriority::Late);

    let mut out: Vec<ItemPriorityGroup> = Vec::new();
    for &priority in &priorities.keys {
        out.push(ItemPriorityGroup {
            priority,
            items: vec![],
        });
    }
    let mut items: Vec<Item> = item_priorities.keys().cloned().collect();
    items.sort();
    for item in items {
        let p = item_priorities[&item];
        let i = priorities.index_by_key[&p];
        out[i].items.push(format!("{:?}", item));
    }
    println!("{:?}", out);
    out
}
