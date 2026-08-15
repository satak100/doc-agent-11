# Knowledge-base pipeline

```mermaid
flowchart LR
    A[Drive: 310 JPEGs] --> B[Ingest: IDs + EXIF]
    B --> C[Layout: page + spotting polygons]
    C --> D[PaddleOCR-VL 1.6]
    D --> E{Token cap?}
    E -- no --> F[Clean text]
    E -- yes --> G[Selective retry + tail trim]
    F --> H[PII redaction]
    G --> H
    H --> I[NFKC + Bangla/Latin script ID]
    I --> I2[64-word chunks, overlap 16]
    I2 --> J1[E5-small 384-D]
    I2 --> J2[Feature hash 2048-D]
    J1 --> K1[Exact cosine index]
    J2 --> K2[Exact cosine index]
    Q[Query] --> L[Dense + lexical search]
    K1 --> L
    K2 --> L
    L --> M[RRF fusion]
    M --> N[Top-k chunks + image verification]
```

OCR is resumable and cached. Failed pages can take a second OCR path using an Rx-anchor crop. After OCR, both final indexes build in about 8.7 seconds. Exact cosine is intentional: 943 chunks are too few to justify approximate search or ANN recall loss.
