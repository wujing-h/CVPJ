import os
import unittest
from collections import Counter

from datasets.ccvid import CCVID
from datasets.move import Move


DATA_ROOT = os.environ.get("CLIP_REID_DATA_ROOT", r"C:\Users\xlc\cvpj\datasets")


class CustomDatasetTest(unittest.TestCase):
    def test_ccvid_parses_video_folders_into_image_samples(self):
        dataset = CCVID(root=os.path.join(DATA_ROOT, "CCVID"), verbose=False)

        self.assertGreater(len(dataset.train), 0)
        self.assertGreater(len(dataset.query), 0)
        self.assertGreater(len(dataset.gallery), 0)

        img_path, pid, camid, trackid = dataset.query[0]
        self.assertTrue(os.path.isfile(img_path))
        self.assertIsInstance(pid, int)
        self.assertIsInstance(camid, int)
        self.assertIsInstance(trackid, int)

    def test_move_builds_deterministic_query_gallery_split_without_train(self):
        dataset = Move(root=os.path.join(DATA_ROOT, "Move"), verbose=False, seed=1)
        dataset_again = Move(root=os.path.join(DATA_ROOT, "Move"), verbose=False, seed=1)

        person_dirs = [
            name for name in os.listdir(os.path.join(DATA_ROOT, "Move"))
            if os.path.isdir(os.path.join(DATA_ROOT, "Move", name))
        ]
        total_images = 0
        for person_dir in person_dirs:
            person_path = os.path.join(DATA_ROOT, "Move", person_dir)
            total_images += len([
                name for name in os.listdir(person_path)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

        self.assertIsNone(dataset.train)
        self.assertEqual(len(dataset.query), len(person_dirs))
        self.assertEqual(len(dataset.gallery), total_images - len(person_dirs))
        self.assertEqual(dataset.query, dataset_again.query)
        self.assertEqual(dataset.gallery, dataset_again.gallery)

        query_counts = Counter(pid for _, pid, _, _ in dataset.query)
        gallery_counts = Counter(pid for _, pid, _, _ in dataset.gallery)
        self.assertTrue(all(count == 1 for count in query_counts.values()))
        self.assertTrue(all(count == 4 for count in gallery_counts.values()))
        self.assertTrue(all(camid == 0 for _, _, camid, _ in dataset.query))
        self.assertTrue(all(camid == 1 for _, _, camid, _ in dataset.gallery))


if __name__ == "__main__":
    unittest.main()
