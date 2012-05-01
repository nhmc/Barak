rm -rf generated
rm -rf _build
mkdir -p generated
./make_index.py
make html
