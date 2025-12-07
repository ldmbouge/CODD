library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# Input

## Read DIPD data
csv_list <- c(
  "../results/tsptw_rpid_t600_AFG_202511270953.csv", 
  "../results/tsptw_rpid_t600_GendreauDumasExtended_202511271054.csv", 
 # "../results/tsptw_rpid_t600_OhlmannThomas_202511271937.csv", 
  "../results/tsptw_rpid_t600_Solnon25_feasible_202511252213.csv", 
  "../results/tsptw_rpid_t600_Solnon25_infeasible_202511260948.csv", 
  "../results/tsptw_rpid_t600_SolomonPesant_202511280050.csv", 
  "../results/tsptw_rpid_t600_SolomonPotvinBengio_202511261103.csv"
)
tmp <- lapply(csv_list, read.csv)
didp_data <- do.call(rbind, tmp)
didp_data <- didp_data %>% mutate(Timeout=tolower(Timeout) == "true")
didp_data$Solver <- "DIDP-RUST"


csv_list <- c(
  "../results/tsptw_didp.py_t600_AFG_202512050732.csv",
  "../results/tsptw_didp.py_t600_GendreauDumasExtended_202512050817.csv",
  "../results/tsptw_didp.py_t600_Solnon25_feasible_202512042030.csv",
  "../results/tsptw_didp.py_t600_Solnon25_infeasible_202512050536.csv",
  "../results/tsptw_didp.py_t600_SolomonPesant_202512050608.csv",
  "../results/tsptw_didp.py_t600_SolomonPotvinBengio_202512050648.csv"
)
tmp <- lapply(csv_list, read.csv)
didp_par_data <- do.call(rbind, tmp)
didp_par_data <- didp_par_data %>% mutate(Timeout=tolower(Timeout) == "true")
didp_par_data$Solver <- "DIDP-PAR"


## Read CADDS-GPU (Unsorted) data
csv_list <- c(
  "../results/tsptw_1_bro_t600_g_AFG_202511262302.csv",
  "../results/tsptw_1_bro_t600_g_GendreauDumasExtended_202511262343.csv", 
  #"../results/tsptw_1_bro_t600_g_OhlmannThomas_202511270542.csv", 
  "../results/tsptw_1_bro_t600_g_Solnon25_feasible_202511261450.csv", 
  "../results/tsptw_1_bro_t600_g_Solnon25_infeasible_202511262151.csv", 
  "../results/tsptw_1_bro_t600_g_s_SolomonPesant_202511281809.csv", 
  "../results/tsptw_1_bro_t600_g_SolomonPotvinBengio_202511262215.csv"
)
tmp <- lapply(csv_list, read.csv)
cadds_data <- do.call(rbind, tmp)
cadds_data <- cadds_data %>% mutate(Timeout=tolower(Timeout) == "true")
cadds_data$Solver <- "CADDS-GPU"

## Read CADDS-SEQ (Unsorted) data
csv_list <- c(
  "../results/tsptw_1_bro_seq_t600_AFG_202512041021.csv",
  "../results/tsptw_1_bro_seq_t600_GendreauDumasExtended_202512041108.csv",
  "../results/tsptw_1_bro_seq_t600_Solnon25_feasible_202512032210.csv",
  "../results/tsptw_1_bro_seq_t600_Solnon25_infeasible_202512040751.csv",
  "../results/tsptw_1_bro_seq_t600_SolomonPesant_202512040820.csv",
  "../results/tsptw_1_bro_seq_t600_SolomonPotvinBengio_202512040928.csv"
)
tmp <- lapply(csv_list, read.csv)
cadds_seq_data <- do.call(rbind, tmp)
cadds_seq_data <- cadds_seq_data %>% mutate(Timeout=tolower(Timeout) == "true")
cadds_seq_data$Solver <- "CADDS-SEQ"

## Read CADDS-GPU (Sorted) data
csv_list <- c(
  "../results/tsptw_1_bro_t600_g_s_AFG_202511281915.csv",
  "../results/tsptw_1_bro_t600_g_s_GendreauDumasExtended_202511281957.csv",
  #"../results/tsptw_1_bro_t600_g_s_OhlmannThomas_202511290151.csv",
  "../results/tsptw_1_bro_t600_g_s_Solnon25_feasible_202511291349.csv",
  "../results/tsptw_1_bro_t600_g_s_Solnon25_infeasible_202511281755.csv",
  "../results/tsptw_1_bro_t600_g_s_SolomonPesant_202511281809.csv",
  "../results/tsptw_1_bro_t600_g_s_SolomonPotvinBengio_202511281833.csv"
)
tmp <- lapply(csv_list, read.csv)
cadds_sorted_data <- do.call(rbind, tmp)
cadds_sorted_data <- cadds_sorted_data %>% mutate(Timeout=tolower(Timeout) == "true")
cadds_sorted_data$Solver <- "CADDS-GPU-SORTED"

## Read RouteOpt data
csv_list <- c(
  "../results/benchmarks_tw_single_t600_AFG_202512021907.csv",                        
  "../results/benchmarks_tw_single_t600_GendreauDumasExtended_202512022307.csv",  
  #"../results/benchmarks_tw_single_t600_OhlmannThomas_202512030924.csv",          
  "../results/benchmarks_tw_single_t600_Solnon25_feasible_202512020129.csv",
  "../results/benchmarks_tw_single_t600_Solnon25_infeasible_202512021254.csv",
  "../results/benchmarks_tw_single_t600_SolomonPesant_202512021906.csv",
  "../results/benchmarks_tw_single_t600_SolomonPotvinBengio_202512021906.csv"
)
tmp <- lapply(csv_list, read.csv)
route_data <- do.call(rbind, tmp)
route_data <- route_data %>% 
  filter(Timeout != "" & !is.na(Timeout)) %>% 
  filter(!(Benchmark == "SolomonPotvinBengio" & Instance == "rc_201.4.txt")) %>% # Solver bugged
  mutate(Timeout=tolower(Timeout) == "true")
route_data$Solver <- "ROUTEOPT"

# Combine all datasets
all_data <- bind_rows(
  didp_data,
  didp_par_data,
  cadds_data,
  cadds_seq_data,
  cadds_sorted_data,
  route_data
)


## Sanity Check across all solvers

# For each instance, check if all non-timeout Best.Cost values match
mismatches <- all_data %>%
  filter(!Timeout) %>%
  group_by(Benchmark, Instance) %>%
  summarise(
    n_solvers = n(),
    unique_costs = n_distinct(Best.Cost),
    costs = paste(unique(Best.Cost), collapse = ", "),
    solvers = paste(Solver, collapse = ", "),
    .groups = "drop"
  ) %>%
  filter(unique_costs > 1)

if (nrow(mismatches) > 0) {
  stop("❌ Mismatched Best.Cost values found:\n", 
       paste(capture.output(print(mismatches)), collapse = "\n"))
} else {
  message("✅ All Best.Cost values match across all solvers (DIDP, CADDS-GPU, CADDS-GPU-SORTED, ROUTEOPT).")
}

# Comparison
names(all_data)

# Step 1: Find active solvers (those who solved at least one instance)
active_solvers <- all_data %>%
  filter(!Timeout) %>%
  select(Benchmark, Solver) %>%
  distinct()

# Step 2: Count how many active solvers per benchmark
n_active_solvers <- active_solvers %>%
  group_by(Benchmark) %>%
  summarise(n_active = n(), .groups = "drop")

# Step 3: Find instances solved by ALL active solvers
common_instances <- all_data %>%
  filter(!Timeout) %>%
  group_by(Benchmark, Instance) %>%
  summarise(n_solvers_solved = n_distinct(Solver), .groups = "drop") %>%
  left_join(n_active_solvers, by = "Benchmark") %>%
  filter(n_solvers_solved == n_active) %>%
  select(Benchmark, Instance)

# Step 4: Find specific instances (NOT solved by all active solvers)
specific_instances <- all_data %>%
  anti_join(common_instances) %>%
  select(Benchmark, Instance) %>%
  distinct()

# Create all possible Solver-Benchmark combinations
all_combinations <- all_data %>%
  select(Benchmark) %>%
  distinct() %>%
  cross_join(all_data %>% select(Solver) %>% distinct())

# Step 5: Statistics for COMMON instances
stats_common <- all_data %>%
  inner_join(common_instances, by = c("Benchmark", "Instance")) %>%
  filter(!Timeout) %>%
  group_by(Benchmark,Solver) %>%
  summarise(
    n_solved = n(),
    # Time statistics
    sum_time = sum(Search.Time),
    min_time = min(Search.Time),
    max_time = max(Search.Time),
    mean_time = mean(Search.Time),
    stddev_time = sd(Search.Time),
    geom_mean_time = exp(mean(log(Search.Time))),
    
    # Memory statistics
    sum_memory = sum(Memory),
    min_memory = min(Memory),
    max_memory = max(Memory),
    mean_memory = mean(Memory),
    stddev_memory = sd(Memory),
    geom_mean_memory = exp(mean(log(Memory))),
    
    # Nodes statistics
    sum_nodes = sum(Nodes),
    min_nodes = min(Nodes),
    max_nodes = max(Nodes),
    mean_nodes = mean(Nodes),
    stddev_nodes = sd(Nodes),
    geom_mean_nodes = exp(mean(log(Nodes))),
    .groups = "drop"
  ) %>%
  # Add ALL combinations (including inactive solvers)
  right_join(all_combinations, by = c("Solver", "Benchmark")) %>%
  arrange(Benchmark,Solver)

# Step 6: Statistics for SPECIFIC instances
stats_specific <- all_data %>%
  inner_join(specific_instances, by = c("Benchmark", "Instance")) %>%
  filter(!Timeout) %>%
  group_by(Benchmark,Solver) %>%
  summarise(
    n_solved = n(),
    # Time statistics
    sum_time = sum(Search.Time),
    min_time = min(Search.Time),
    max_time = max(Search.Time),
    mean_time = mean(Search.Time),
    stddev_time = sd(Search.Time),
    geom_mean_time = exp(mean(log(Search.Time))),
    
    # Memory statistics
    sum_memory = sum(Memory),
    min_memory = min(Memory),
    max_memory = max(Memory),
    mean_memory = mean(Memory),
    stddev_memory = sd(Memory),
    geom_mean_memory = exp(mean(log(Memory))),
    
    # Nodes statistics
    sum_nodes = sum(Nodes),
    min_nodes = min(Nodes),
    max_nodes = max(Nodes),
    mean_nodes = mean(Nodes),
    stddev_nodes = sd(Nodes),
    geom_mean_nodes = exp(mean(log(Nodes))),
    .groups = "drop"
  ) %>%
  # Add ALL combinations (including inactive solvers)
  right_join(all_combinations, by = c("Solver", "Benchmark")) %>%
  arrange(Benchmark,Solver)

# Rounds and save to CSV
temp <- stats_common %>%
  mutate(across(where(is.numeric), ~round(., 2)))
write.csv(temp, "stats_common_instances.csv", row.names = FALSE)

temp <- stats_specific %>%
  mutate(across(where(is.numeric), ~round(., 2)))
write.csv(temp, "stats_specific_instances.csv", row.names = FALSE)

