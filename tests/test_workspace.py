import tempfile
import unittest
from pathlib import Path

from src.rag.workspace import list_pdfs, read_json, save_pdf, validate_url, write_json, remove_pdf, restore_pdf


class WorkspaceTests(unittest.TestCase):
    def test_remove_restore_and_conflicting_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "data/pdfs/nested"
            original = save_pdf(library, "one.pdf", b"%PDF-1.4\noriginal")
            archived = remove_pdf(root, original)
            self.assertFalse(list_pdfs(root / "data/pdfs"))
            self.assertEqual(archived.read_bytes(), b"%PDF-1.4\noriginal")
            save_pdf(library, "one.pdf", b"%PDF-1.4\nnew")
            with self.assertRaises(FileExistsError):
                restore_pdf(root, archived)
            self.assertTrue(archived.exists())
            remove_pdf(root, original)
            restored = restore_pdf(root, archived)
            self.assertEqual(restored.resolve(), original.resolve())
            self.assertEqual(restored.read_bytes(), b"%PDF-1.4\noriginal")

    def test_remove_rejects_outside_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outside = save_pdf(root, "outside.pdf", b"%PDF-1.4\n")
            library = root / "data/pdfs"
            library.mkdir(parents=True)
            link = library / "link.pdf"
            link.symlink_to(outside)
            for path in (outside, link):
                with self.assertRaises(ValueError):
                    remove_pdf(root, path)
                with self.assertRaises(ValueError):
                    restore_pdf(root, path)
            self.assertTrue(outside.exists())

    def test_upload_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_pdf(root, "one.pdf", b"%PDF-1.4\n")
            with self.assertRaises(FileExistsError):
                save_pdf(root, "one.pdf", b"%PDF-1.4\nchanged")
            for name in ("../outside.pdf", "nested/file.pdf", "bad\\file.pdf"):
                with self.assertRaises(ValueError):
                    save_pdf(root, name, b"%PDF-1.4\n")
            with self.assertRaises(ValueError):
                save_pdf(root, "fake.pdf", b"plain text")
            self.assertEqual(len(list_pdfs(root)), 1)

    def test_urls(self):
        self.assertEqual(validate_url("https://youtu.be/dQw4w9WgXcQ?t=12", "youtube"),
                         "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        for url in ("http://example.com", "https://user:pass@example.com", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                validate_url(url, "websites")
        with self.assertRaises(ValueError):
            validate_url("https://example.com/watch?v=dQw4w9WgXcQ", "youtube")

    def test_settings_preserve_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            example = Path(folder) / "example.json"
            actual = Path(folder) / "settings.json"
            write_json(example, {"model": "original"})
            settings = read_json(actual, example)
            settings["model"] = "changed"
            write_json(actual, settings)
            self.assertEqual(read_json(actual, example)["model"], "changed")
            self.assertEqual(read_json(example, example)["model"], "original")


if __name__ == "__main__":
    unittest.main()
