import os
import tempfile

os.environ.setdefault("LDX_HOME", tempfile.mkdtemp(prefix="ldx_test_"))   # tests must never touch real data
