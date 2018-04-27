#!/bin/bash -e

usage() {
  echo 'usage: docs-anonymiser.sh [-d directory] | [-h]'
  echo ''
  echo ' -d, --directory              Directory name containning documents to be anonymised'
  echo ' -s, --skewness-correction    Anonymise PDFs with skewness correction enabled (e.g. from scanned PDFs)'
  echo ' -h, --help                   Prints this message'
  echo ''
  echo ' docs-anonymiser.sh -d ~/Documents/PDFs'
  exit 1
}

[[ -z "$1" ]] && usage && exit 1

while test $# -gt 0; do
  case $1 in
    -d | --directory)
        shift
        [[ -z "$1" ]] && usage || DIR=$1
        ;;
    -s | --skewness-correction)
        SKEWNESS="1"
        ;;
    -h | --help)
        usage
        ;;
    * )

        usage
        ;;
  esac
  shift
done

host_dir=$(realpath $DIR)
workdir=$(grep WORKDIR Dockerfile | awk '{print $2}')
echo "Creating anonymised files in: $host_dir/Anonymised"

image_name="quay.io/wealthwizards/docs-anonymiser"
release=$(curl -s https://api.github.com/repos/WealthWizardsEngineering/docs-anonymiser/releases/latest |grep tag_name |cut -d \" -f 4)
command="$workdir $SKEWNESS"

sudo -n docker run --rm -i \
  -v $host_dir:$workdir:Z \
  $image_name:$release anonymise.py $command
