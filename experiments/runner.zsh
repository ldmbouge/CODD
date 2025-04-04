#!/bin/zsh

# for f in ../data/knapsack/knapPI_*; do
#     fname=$(basename $f)
#     part="${fname#knapPI_}"
#     if [[ ("$part" = "3_10000_1000_1") || ("$part" = "3_2000_1000_1") || ("$part" = "3_5000_1000_1")]]; then
#         for w in 256 512 1024 2048 4096; do
#             for s in XFLD RFLD; do
#                 (echo $f w:$w search:$s; ./knapsack $f $w $s | tail -1)
#             done
#         done
#     else
#         for w in 16 32 64 128 256 512; do
#             for s in XFLD RFLD; do
#                 (echo $f w:$w search:$s; ./knapsack $f $w $s | tail -1)
#             done
#         done
#     fi
# done

# for f in ../data/knapsack/knapPI_*; do
#     fname=$(basename $f)
#     part="${fname#knapPI_}"
#     if [[ ("$part" = "3_10000_1000_1") || ("$part" = "3_2000_1000_1") || ("$part" = "3_5000_1000_1")]]; then
#         for w in 256 512 1024 2048 4096; do
#             (echo $f w:$w; ./knapsack $f $w XFLD | tail -1)
#         done
#     else
#         for w in 16 32 64 128 256 512; do
#             (echo $f w:$w; ./knapsack $f $w XFLD | tail -1)
#         done
#     fi
# done

#for f in johnson8-2-4 johnson8-4-4 johnson16-2-4 keller4 hamming6-2 hamming6-4 hamming8-2 hamming8-4 hamming10-2 brock200_1 brock200_2 brock200_3 brock200_4 p_hat300-1 p_hat300-2; do
# for f in hamming8-4 hamming10-2 brock200_1 brock200_2 brock200_3 brock200_4 p_hat300-1 p_hat300-2; do
#     (echo $f w:128 $(date +%d.%m.%y-%H:%M:%S); ./misp5RF ../data/misp/$f.clq 128 | tail -1)
# done

# for f in myciel3 myciel4 2-Insertions_3 1-FullIns_3 2-FullIns_3 r125.1 r125.1c anna jean david queen5_5 queen6_6 queen7_7 DSJC125.1 miles250 miles500 miles750 school1 school1_nsh ; do
#     (echo $f w:64 $(date +%d.%m.%y-%H:%M:%S); ./coloring2RF ../data/coloring/DIMACS/$f.col 64 | tail -1)
# done

# for fname in ../data/tsptw/**/*.*; do

#     dir=$(dirname "$fname")
#     if [[ "$skip_dir" == "$dir" ]]; then
#         continue
#     fi

#     for w in 256 1024; do
#         echo $fname w:$w $(date +%d.%m.%y-%H:%M:%S)
#         output=$(gtimeout 1000 sh -c "./tsptw \"$fname\" \"$w\" | tail -1")
#         if (($?)); then
#             echo "T.O."
#             skip_dir="$dir"
#         else
#             echo "$output"
#         fi
#     done
# done

for fname in ../data/tsptw/AFG/rbg010a.tw ../data/tsptw/AFG/rbg016a.tw ../data/tsptw/AFG/rbg016b.tw ../data/tsptw/Langevin/N20ft301.dat ../data/tsptw/Langevin/N20ft302.dat ../data/tsptw/Langevin/N20ft303.dat ../data/tsptw/Langevin/N20ft304.dat ../data/tsptw/Langevin/N20ft305.dat ../data/tsptw/Langevin/N20ft306.dat ../data/tsptw/Langevin/N20ft307.dat ../data/tsptw/Langevin/N20ft308.dat ../data/tsptw/Langevin/N20ft309.dat ../data/tsptw/Langevin/N20ft310.dat ../data/tsptw/Langevin/N20ft401.dat ../data/tsptw/Langevin/N20ft402.dat ../data/tsptw/Langevin/N20ft403.dat ../data/tsptw/Langevin/N20ft404.dat ../data/tsptw/Langevin/N20ft405.dat ../data/tsptw/Langevin/N20ft406.dat ../data/tsptw/Langevin/N20ft407.dat ../data/tsptw/Langevin/N20ft408.dat ../data/tsptw/Langevin/N20ft409.dat ../data/tsptw/Langevin/N20ft410.dat ../data/tsptw/SolomonPotvinBengio/rc_201.1.txt ../data/tsptw/toy/toy1.tw ../data/tsptw/toy/toy2.tw; do
    for w in 256 1024; do
        echo $fname w:$w $(date +%d.%m.%y-%H:%M:%S)
        output=$(gtimeout 100 sh -c "./tsptwRF \"$fname\" \"$w\" | tail -1")
        if (($?)); then
            echo "T.O."
            skip_dir="$dir"
        else
            echo "$output"
        fi
    done
done
# ./knapsack ../data/knapsack/knapPI_3_5000_1000_1 32 XFLD > XFLD_3_5000_1000
# ./knapsack ../data/knapsack/knapPI_3_5000_1000_1 32 RFLD > RFLD_3_5000_1000