# Freeze addendum (2026-11-13)

`freeze-manifest.json` is an immutable record of the Pilot 4 freeze and was not modified. Its SHA-256 is `2c16a8a56117673967c2226f35709a16ad293342b08d40a5aa7ea13678be55f8`.

35 of its 36 file entries match the files on disk. The one mismatch is deliberate:

| File | Hash at freeze | Hash now |
|---|---|---|
| `pilot3/manifest.json` | `d5b71c2ba64daaf8cc912077e8885f9252de3acb276cdfc3579b6182888accc5` | `d39b5de8dda659b19a5cbab90c772f9bdf6def416da4f2563ce533aab9e292f0` |

Reason: reviewer corrections on 2026-11-13. `pilot3/RESULTS.md` was corrected (wording only; no number or verdict changed), so its entry in `pilot3/manifest.json` was updated from `92aec9b5a36ff66f617164f9d7e2139c615bf84bded5d8e7e89331c6286a91dc` to `5924fb89e2154c9184e2c1640398908d0c866efdbbe85628812de598030cf80f`, and a `changelog` array recording both hashes was added. `pilot4/DATA_AUDIT.md` was also corrected (hash now `2e0a0f24aee7901e5c0cbc8a2935a188437ceb803b4fe3b7669c23721cd39879`); it is not listed in the freeze manifest.
