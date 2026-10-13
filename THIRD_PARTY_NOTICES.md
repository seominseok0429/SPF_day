# Third-party attribution

The model and flow scheduler are runtime dependencies of
[stanfordmlgroup/spf](https://github.com/stanfordmlgroup/spf), commit
`a7624ae25b47b474d5a31ca22b3ac6e6907795dd`, distributed under Apache-2.0.
Its license is preserved in `third_party_licenses/SPF_Apache-2.0.txt`.
No upstream full source tree or pretrained weights are vendored here.

The model adapter modifies the runtime forward gradient context as documented
in `spf_ssp126/model.py`. This does not edit the upstream checkout.
Project-specific flow/auxiliary-loss scripts were reorganized for SSP126-only
training and portable command-line usage on 2026-10-13.

Static AR6 masks, climate data, official indices/scorer and historical threshold
inputs remain external and subject to their own licenses. No evaluation target
values or fitted calibration parameters are distributed.
