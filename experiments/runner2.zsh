for fname in ../data/tsptw/**/*.*; do
    if [[ "$fname" == *.yaml ]]; then
        continue
    fi

    for solver in "" 2 2RF 3 3RF 3RFT 3RO 3RONQ RF RO RONQ; do
        for w in 64 256 1024; do
            echo tsptw_triangle$solver $fname $w $(date +%d.%m.%y-%H:%M:%S)
            output=$(gtimeout 1 sh -c "../build/tsptw_triangle$solver \"$fname\" \"$w\" | tail -1")
            if (($?)); then
                echo "T.O."
            else
                echo "$output"
            fi
        done
    done
done