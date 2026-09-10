# Upstream references and attribution

The additional method integrations are independent continuous PyTorch
implementations of the cited algorithmic mechanisms. They do not bundle or
import the inspected upstream repositories. Exact inspected revisions and
paper/source URLs are in [UPSTREAM_REVISIONS.json](UPSTREAM_REVISIONS.json).
That manifest also lists every material implementation substitution.

Credit for the algorithms belongs to their paper authors and upstream
maintainers. Preserve their citations when using these method names in a
research comparison. Known upstream MIT notices are retained here as attribution:

- [Design-Baselines / MINs](upstream-licenses/mins.txt)
- [GABO](upstream-licenses/gabo.txt)
- [ROOT](upstream-licenses/root.txt)
- [SPADE](upstream-licenses/spade.txt)

No upstream weights, datasets, or downloaded source snapshots are redistributed
with these integrations. Data Recipes continues to be fetched independently at
the recorded commit into the user's external-data directory or Docker volume.
