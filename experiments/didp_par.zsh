#!/bin/bash
maxJobs=90
export PATH=$PATH:$HOME/.cargo/bin

doIt() {
    id=$1
    name=$2
    w="_"
    echo $name w:$w $(date +%d.%m.%y-%H:%M:%S) > $id.$w.res
    output=$(timeout 300s sh -c "didp-yaml didp-models/tsptw/domain_non_zero_base.yaml $name didp-models/configs/cabs.yaml | tail -4 | cut -d: -f2  | xargs | sed 's/s//g'")
    if (($?)); then
        echo "T.O." >> $id.$w.res
    else
        echo "$output" >> $id.$w.res
    fi    
}
export -f doIt

for folder in tsptw/*; do
    nb=`ls -l $folder/*.yaml | wc -l`
    files=`find $folder -type f -name '*.yaml' | sort | xargs` 
    echo Folder \[$folder\] $nb
    echo \[$files\]
    SHELL=$(type -p bash) parallel -j$maxJobs echo Job {#} {1} \; doIt {#} {1}  ::: $files
    nbJobs=`ls -l *.res | wc -l`
    echo "Number of jobs: $nbJobs"
    for f in `seq 1 $nbJobs`; do
	cat  $f.*.res >> result.$(basename $folder).dat
    done
    rm *.res
done

