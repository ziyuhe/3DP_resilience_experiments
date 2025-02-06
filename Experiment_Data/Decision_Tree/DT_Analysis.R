# Install required packages
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("caret")

# Load required libraries
# library(rpart)
# library(rpart.plot)
# library(caret)


#########################################################################################################
# Data Preparation
#########################################################################################################

# Read the data (adjust i and j based on your scenario)
i <- 1  # specify your i
j <- 1  # specify your j
filename_base <- sprintf("scenario_speed_%d_cost_%d", i, j)

# Read features and responses (using relative paths)
raw_features <- read.csv(paste0(filename_base, "_raw_features.csv"))
synth_features <- read.csv(paste0(filename_base, "_synth_feature.csv"))
responses <- read.csv(paste0(filename_base, "_responses.csv"))

# Print available features for selection
cat("Available Raw Features:\n")
print(names(raw_features))
cat("\nAvailable Synthetic Features:\n")
print(names(synth_features))


# Rename features in synth_features
target_names <- c("Relative.3DP.profit", "mean.shortfall")
new_names <- c("3DP Profit-to-Cost", "Mean Shortfall")

# Change column names in synthetic features
names(synth_features)[names(synth_features) == target_names[1]] <- new_names[1]
names(synth_features)[names(synth_features) == target_names[2]] <- new_names[2]

# Overwrite "3DP Profit-to-Cost" with its reciprocal
synth_features$`3DP Profit-to-Cost` <- 1 / synth_features$`3DP Profit-to-Cost`
###################################################################################################






###################################################################################################
# FIRST TRY ALL THE SYNTHETIC FEATURES AND CHECK THE IMPORTANCE SCORES FOR FEATURE SELECTION
###################################################################################################
# Try synthetic features
selected_synth_features <- c("3DP Profit-to-Cost", "service.level1", "service.level2", "service.level3", "Mean Shortfall", "DB.Retainer.Rate")
# selected_synth_features <- c("3DP Profit-to-Cost", "Mean Shortfall")

# Combine selected features
X <- cbind(
  synth_features[, selected_synth_features, drop=FALSE]
)
y <- responses$response

# Convert response variable to a factor with descriptive labels
y <- factor(y, levels = c(0, 1), labels = c("No Switch", "Switched to 3DP"))

#########################################################################################################


#########################################################################################################
# NOW WE DO K-FOLD CV ON "Max Splits" and "Max Depth"
#########################################################################################################
# Create splits vector
splits <- c(1:4, seq(5, 200, by=10))
depths <- c(1:5, 10, 20,30)

# Number of folds for cross-validation
k <- 5

# Initialize matrix to store results
results <- matrix(0, nrow=length(splits), ncol=length(depths))

# Perform grid search with k-fold cross validation
total_iterations <- length(depths) * length(splits)
current_iteration <- 0

for (d in 1:length(depths)) {
  for (s in 1:length(splits)) {

    current_iteration <- current_iteration + 1
    progress <- (current_iteration/total_iterations) * 100

    cat(sprintf("Progress: %.1f%%, Split = %d, Depth = %d\n",
                progress, splits[s], depths[d]))

    # Initialize vector to store k-fold accuracies
    fold_accuracies <- numeric(k)

    # Create k folds
    folds <- createFolds(y, k=k, list=TRUE)

    # Perform k-fold cross validation
    for (fold in 1:k) {
      # Split data into training and validation
      train_indices <- unlist(folds[-fold])
      valid_indices <- unlist(folds[fold])

      # Train model
      tree <- rpart(y ~ .,
                    data=data.frame(X, y=factor(y))[train_indices,],
                    control=rpart.control(maxdepth=depths[d],
                                          maxcompete=0,
                                          maxsurrogate=0,
                                          minsplit=2,
                                          cp=1e-6,  # very small complexity parameter
                                          maxnode=splits[s] + 1))  # splits + 1 = number of nodes

      # Make predictions
      pred <- predict(tree, data.frame(X)[valid_indices,], type="class")

      # Calculate accuracy
      fold_accuracies[fold] <- mean(pred == y[valid_indices])
    }

    # Store average accuracy
    results[s,d] <- mean(fold_accuracies)
  }
}

#########################################################################################################



#########################################################################################################
# PRINT THE ACCURACY RESULTS UNDER DIFFERENT "Max Splits" and "Max Depth"
#########################################################################################################
# Create a labeled matrix/table of results
results_table <- as.data.frame(results)
# Set column names as depths
colnames(results_table) <- paste("Depth", depths)
# Set row names as splits
rownames(results_table) <- paste("Split", splits)

# Print the table
print(results_table)
#########################################################################################################



#########################################################################################################
# CHOOSE A GOOD PAIR OF HYPER-PARAMETERS, RE-TRAIN AND PLOT THE TREE
#########################################################################################################
# Assuming you've chosen your splits and depth
chosen_splits <- 3  
chosen_depth <- 4   

# Train the final tree
final_tree <- rpart(y ~ ., 
                    data=data.frame(X, y=factor(y)),
                    control=rpart.control(maxdepth=chosen_depth,
                                          maxcompete=0,
                                          maxsurrogate=0,
                                          minsplit=2,
                                          cp=1e-6,
                                          maxnode=chosen_splits + 1
                    ))

# Get variable importance
importance <- final_tree$variable.importance

# Create a data frame with feature names and their importance
importance_df <- data.frame(
  Feature = names(importance),
  Importance = importance
)

# Sort by importance and print
importance_df <- importance_df[order(-importance_df$Importance), ]
print(importance_df)

# NOTE: Only "3DP.Profit.to.Cost" and "Mean.Shortfall" have significant "Feature Importance"
#########################################################################################################













#########################################################################################################
# Re-train the model with only "3DP.Profit.to.Cost" and "Mean.Shortfall"
#########################################################################################################

selected_synth_features <- c("3DP Profit-to-Cost", "Mean Shortfall")

# Combine selected features
X <- cbind(
  synth_features[, selected_synth_features, drop=FALSE]
)
y <- responses$response

# Convert response variable to a factor with descriptive labels
y <- factor(y, levels = c(0, 1), labels = c("No Switch", "Switched to 3DP"))

# Assuming you've chosen your splits and depth
chosen_splits <- 3 
chosen_depth <- 4   

# Train the final tree
final_tree <- rpart(y ~ ., 
                    data=data.frame(X, y=factor(y)),
                    control=rpart.control(maxdepth=chosen_depth,
                                          maxcompete=0,
                                          maxsurrogate=0,
                                          minsplit=2,
                                          cp=1e-6,
                                          maxnode=chosen_splits + 1
                    ))


# The feature names are all screwed up, we change them manually
final_tree$frame$var <- sub("\\.", " ", final_tree$frame$var)
final_tree$frame$var <- gsub("\\.", "-", final_tree$frame$var)
final_tree$frame$var <- gsub("^X", "", final_tree$frame$var)

# Save the decision tree plot to a PNG file
pdf("final_decision_tree_full.pdf", width = 12, height = 8)  # Set PDF dimensions
rpart.plot(final_tree, type = 5, cex = 1.5)
dev.off()
#########################################################################################################




#########################################################################################################
# IF THE TREE IS TOO MUCH, WE PRUNE IT
#########################################################################################################

# Prune the tree using the optimal cp
pruned_tree <- prune(final_tree, cp=0.006)

# Visualize the pruned tree
rpart.plot(pruned_tree)

pdf("final_decision_tree_pruned.pdf", width = 12, height = 8)  # Set PDF dimensions
rpart.plot(pruned_tree, type = 5, cex = 1.5)
dev.off()
#########################################################################################################
