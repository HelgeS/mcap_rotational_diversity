#!/usr/bin/env bash
rm run.sh

for f in "$@"; do
    echo "/src/$f switch -t 35 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 25 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 15 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 9 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 8 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 7 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 6 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 5 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 4 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 3 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 2 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 1 --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 40 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 30 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 20 --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 10 --timeout 60 -o " >> run.sh
	echo "/src/$f wpp --ind-weights --timeout 60 -o " >> run.sh
	echo "/src/$f profit --timeout 60 -o " >> run.sh
	echo "/src/$f affinity --timeout 60 -o " >> run.sh
	echo "/src/$f productcomb --timeout 60 -o " >> run.sh

    echo "/src/$f switch -t 35 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 25 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 15 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 9 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 8 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 7 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 6 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 5 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 4 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 3 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 2 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 1 --limit-assignments --timeout 60 -o " >> run.sh
    echo "/src/$f switch -t 40 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 30 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 20 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f switch -t 10 --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f wpp --ind-weights --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f profit --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f affinity --limit-assignments --timeout 60 -o " >> run.sh
	echo "/src/$f productcomb --limit-assignments --timeout 60 -o " >> run.sh
done
