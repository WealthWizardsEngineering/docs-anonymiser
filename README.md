## Requirements

- [Docker](https://docs.docker.com/docker-for-mac/install/)
- Bash


## To run

For PDFs that have been digitally generated, place all PDFs in a folder and run

`./docs-anonymiser.sh -d folder_path`

The character recognition will not perform well if the pages are skewed (e.g. from scanned PDFs) so they need to be corrected. Performing skewness correction on pages that do not require it, will not change them but it will increase running time (~1 min per page). To anonymise PDFs with skewness correction run

`./docs-anonymiser.sh -d folder_path -s`

## License

Copyright (C) 2018 Wealth Wizards, Ltd. Distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
