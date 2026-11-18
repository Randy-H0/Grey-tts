# Security

Found a security problem (for example in the model downloader, a way to strip or forge the AI watermark, or anything
that could harm users)? Please **don't open a public issue**. Email **support@thisvoiceisai.com** with the details and
we'll get back to you.

Model files are checked against their SHA-256 before they are used (`download_model.py`). Only load `.pt` files you
trust: PyTorch checkpoints can contain code.
