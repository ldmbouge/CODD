library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(tikzDevice)

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
  filter(!(Benchmark == "SolomonPotvinBengio" & Instance == "rc_201.4.txt")) %>% # Solver bugged
  mutate(Timeout=tolower(Timeout) == "true")
route_data$Solver <- "ROUTEOPT"

## Read CODD-BB data
csv_list <- c(
  "../results/tsptw_mst_triangleRO1_t600_AFG_202512081615.csv",
  "../results/tsptw_mst_triangleRO1_t600_GendreauDumasExtended_202512082057.csv",
  "../results/tsptw_mst_triangleRO1_t600_Solnon25_feasible_202512071458.csv",
  "../results/tsptw_mst_triangleRO1_t600_Solnon25_infeasible_202512081021.csv",
  "../results/tsptw_mst_triangleRO1_t600_SolomonPesant_202512081443.csv",
  "../results/tsptw_mst_triangleRO1_t600_SolomonPotvinBengio_202512091812.csv"

)
tmp <- lapply(csv_list, read.csv)
codd_bb_data <- do.call(rbind, tmp)
codd_bb_data <- codd_bb_data %>% mutate(Timeout=tolower(Timeout) == "true")
codd_bb_data$Solver <- "CODD-BB-MST"

# Combine all datasets
all_data <- bind_rows(
  didp_data,
  didp_par_data,
  cadds_data,
  cadds_seq_data,
  cadds_sorted_data,
  route_data,
  codd_bb_data
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

filter_solvers_list <- c(
  "DIDP-RUST", 
  "DIDP-PAR",
  "CADDS-GPU", 
  "CADDS-SEQ", 
  "CADDS-GPU-SORTED"
)

display_solvers_list <- c(
  "DIDP-RUST", 
  "DIDP-PAR",
  "CADDS-GPU", 
  "CADDS-SEQ", 
  "CADDS-GPU-SORTED",
  "ROUTEOPT",
  "CODD-BB-MST"
)

solver_order <- c(
  "CADDS-GPU",
  "CADDS-GPU-SORTED",
  "CADDS-SEQ", 
  "DIDP-PAR",
  "DIDP-RUST", 
  "CODD-BB-MST",
  "ROUTEOPT"
)

# Comparison
names(all_data)

all_data <- all_data %>%
  mutate(
    Solver = factor(Solver, levels = solver_order)
  )

benchmarks_info <- all_data %>%
  select(Benchmark, Instance) %>%
  distinct() %>%
  group_by(Benchmark) %>%
  summarise(
    n_instances = n(),
    .groups = "drop"
  )

common_instances <- all_data %>% 
  filter(!Timeout) %>% 
  group_by(Benchmark, Instance) %>% 
  filter(all(filter_solvers_list %in% Solver)) %>%  # every solver appears at least once
  select(Benchmark, Instance) %>%
  distinct()

# Step 4: Find specific instances (NOT solved by all active solvers)
specific_instances <- all_data %>%
  anti_join(common_instances) %>%
  select(Benchmark, Instance) %>%
  distinct()

# Step 5: Statistics for COMMON instances
# Statistics for COMMON instances (using display solvers only)
stats_common <- all_data %>%
  filter(!Timeout) %>%
  semi_join(common_instances, by = c("Benchmark", "Instance")) %>%
  group_by(Benchmark, Solver) %>%
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
  complete(Solver = display_solvers_list, Benchmark) %>%
  arrange(Benchmark, factor(Solver, levels = solver_order))

# Rounds and save to CSV
temp <- stats_common %>%
  mutate(
    across(ends_with("_nodes"), ~./1e6),           # All node columns
    across(ends_with("_memory"), ~./(1024*1024))       # All memory columns
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  mutate(Benchmark = ifelse(duplicated(Benchmark), "", Benchmark))
write.csv(temp, "stats_common_instances.csv", row.names = FALSE,quote = FALSE, na = "")

temp <- stats_specific %>%
  group_by(Benchmark) %>%
  filter(sum(n_solved, na.rm = TRUE) > 0) %>%  # Keep only benchmarks with at least 1 solved instance
  ungroup() %>%
  mutate(
    across(ends_with("_nodes"), ~./1e6),           # All node columns
    across(ends_with("_memory"), ~./(1024*1024))       # All memory columns
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  mutate(Benchmark = ifelse(duplicated(Benchmark), "", Benchmark))
write.csv(temp, "stats_specific_instances.csv", row.names = FALSE,quote = FALSE, na = "")


# Set timeout value
timeout_value <- 600

normalize_solver <- function(x) {
  case_when(
    x == "DIDP-RUST"        ~ "RIDP",
    x == "DIDP-PAR"         ~ "DIDP-Parallel",
    x == "CADDS-GPU"        ~ "CADDS-GPU",
    x == "CADDS-SEQ"        ~ "CADDS-Sequential",
    x == "CADDS-GPU-SORTED" ~ "CADDS-GPU-Greedy",
    x == "ROUTEOPT"         ~ "RouteOpt",
    x == "CODD-BB-MST"      ~ "CODD-BnB-MST",
    TRUE                    ~ x
  )
}

normalize_benchmark <- function(x) {
  case_when(
    x == "Solnon25_feasible"     ~ "Solnon (Feasible)",
    x == "Solnon25_infeasible"   ~ "Solnon (Infeasible)",
    x == "SolomonPotvinBengio"   ~ "Solomon-Potvin-Bengio",
    x == "GendreauDumasExtended" ~ "Gendreau-Dumas-Extended",
    x == "SolomonPesant"         ~ "Solomon-Pesant",
    TRUE                         ~ x
  )
}

solver_colors <- c(
  "RIDP"             = "#d73027",  # red (Rust)
  "DIDP-Parallel"    = "#fdd835",  # yellow (Python style)
  "CADDS-GPU"        = "#1b7837",  # dark green
  "CADDS-Sequential" = "#0071c5",  # Intel blue
  "CADDS-GPU-Greedy" = "#a6d96a",  # light green
  "RouteOpt"         = "#984ea3",  # purple
  "CODD-BnB-MST"     = "#00bfc4"   # cyan
)

solver_linetypes <- c(
  "RIDP"             = "31",      # solid line
  "DIDP-Parallel"    = "31",         # 4 on, 2 off
  "CADDS-GPU"        = "solid",         # 1 on, 3 off (dotted)
  "CADDS-Sequential" = "solid",       # 1 on, 3 off, 4 on, 3 off
  "CADDS-GPU-Greedy" = "solid",         # 7 on, 3 off (long dash)
  "RouteOpt"         = "11",        # 2 on, 2 off, 6 on, 2 off
  "CODD-BnB-MST"     = "1131"   # cyan
)

solver_order_norm <- normalize_solver(solver_order)

performance_data <- all_data %>%
  select(Benchmark, Solver, Instance, Search.Time, Timeout) %>%
  mutate(
    Search.Time = ifelse(is.na(Search.Time) | Timeout, timeout_value * 2, Search.Time),
    Solver      = normalize_solver(Solver),
    Solver      = factor(Solver, levels = solver_order_norm),
    Solver_plot = factor(Solver, levels = rev(solver_order_norm)),
    Benchmark   = normalize_benchmark(Benchmark)
  )



# Plot with stat_ecdf - now with both color and linetype aesthetics
p <- ggplot(performance_data, aes(x = Search.Time, color = Solver_plot, linetype = Solver_plot)) +
  stat_ecdf(geom = "step", linewidth = 0.75, pad = TRUE) +
  facet_wrap(~ Benchmark, ncol = 2) +
  coord_cartesian(xlim = c(0, timeout_value)) +
  scale_y_continuous(limits = c(0.5, 1), labels = scales::label_percent(suffix = "")) +
  labs(x = "Time [s]", 
       y = "Solved Instances [\\%]",
       color = "Solver",
       linetype = "Solver") +
  scale_color_manual(
    values = solver_colors,
    breaks = levels(performance_data$Solver_plot),
    labels = levels(performance_data$Solver_plot)
  ) +
  scale_linetype_manual(
    values = solver_linetypes,
    breaks = levels(performance_data$Solver_plot),
    labels = levels(performance_data$Solver_plot)
  ) +
  guides(
    color = guide_legend(reverse = TRUE),
    linetype = guide_legend(reverse = TRUE)
  ) +
  theme_minimal() +
  theme(
    legend.position = c(0.322, 0.828),
    legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
    legend.text = element_text(size = 6),
    legend.title = element_blank(),
    legend.key.width = unit(14, "pt"),
    legend.key.height = unit(8, "pt"),
    legend.key.spacing.x = unit(0, "pt"),
    legend.key.spacing.y = unit(0, "pt"),
    legend.spacing.y = unit(0, "pt"),
    legend.spacing.x = unit(0, "pt"),
    legend.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt"),
    plot.margin = margin(0, 0, 0, 0),
    panel.grid.minor = element_blank()
  )
p
k <- 2.5
h <-2.2
w <-2
ggsave("ecdf_plot.pdf", plot = p, width = k * w, height = k*h, units = "in", dpi = 300)
tikz("ecdf_plot.tikz", width = k * w, height = k*h, standAlone = FALSE)
print(p)
dev.off()

