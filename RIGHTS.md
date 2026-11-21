# Source and fixture provenance

The Python package is a custom independent reconstruction developed in the
project owner's prior ChatGPT work and extended here. Public comparator
repositories informed the method comparison; no upstream implementation code,
weights, real aircraft data, or third-party media is included. The repository
does not designate a blanket redistribution or open-source license for the
program.

Both bundled WAVs and every input row in the two fixture sets were generated
algorithmically. They contain no recorded voices, real flights, actual
locations, or geographical basemaps. The original small synthetic test fixture
was documented as CC0 1.0 in the prior package; that applies to that fixture,
not to the program or to third-party sources. The wide signed-PCM WAV is the
exact generated input for the demonstrated 27-minute run.

The field replay links to [Eclipse Soundscapes record 21808323](https://doi.org/10.5281/zenodo.21808323),
archived by Henry Winter, MaryKay Severino, and the volunteer scientist, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its WAV excerpts are
generated locally, with samples copied losslessly into a new WAV header. The
original recording remains at Zenodo. The field replay obtains its historical
tracks from [adsb.lol contributors](https://github.com/adsblol/globe_history_2024)
under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The
[replay protocol](benchmarks/eclipse-505-protocol.md) documents how to derive
the exact regional flight cut from the public release. The field figure uses
the public-domain [USGS The National Map topo service](https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer)
and depicts derived spectral intensity and sampled fit residuals.

The standalone result map uses caller-supplied geometry and derived solver scores.
Its optional browser basemap comes from the same USGS topographic tile service.
No basemap tiles are bundled or fetched by the CLI.

The [Livermore input verifier](benchmarks/livermore-inputs.md) acquires the
two original WAVs from Praveen Pathak, *Aeroplane data full analysis and
raw files*, v2 [Zenodo 24075029](https://doi.org/10.5281/zenodo.24075029),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It also
checks historical rows attributed to [adsb.lol contributors](https://www.adsb.lol/docs/open-data/historical)
under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) via
the [adsb.exposed public index](https://github.com/ClickHouse/adsb.exposed#data-source).
The repo contains an acquisition and verification recipe and aggregate
checksums. The clips are downloaded into the ignored `runs/` directory.
The checker emits only track counts and checksums. The linked
DopplerTrack result is a separate comparator with supplied tracks and
timing, credited in the input guide.
