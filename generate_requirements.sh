# generates requirements.txt using pipreqs tool

pip install pipreqs
pipreqs . --mode no-pin --savepath reqs.txt # remove no-pin to get exact versions

mv reqs.txt requirements.txt

# add additional "hidden" dependencies not caught by pipreqs, if necessary
#deps="package1 package2" # packages, space delimited
deps="transformers[torch]"

for dep in $deps
do
  echo $dep >> requirements.txt
done


# to install run:
# pip install -r requirements.txt