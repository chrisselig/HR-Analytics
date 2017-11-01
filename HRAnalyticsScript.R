setwd("G:/Personal/R coding/Rmarkdown/HR_Analytics/HR-Analytics")

#Avoid scientific notation
options(scipen=999)

############################################################ Packages - Other specific packages loaded when needed

require(tidyverse);library(caret)

############################################################ Load Data

rawData <- read_csv("HR_comma_sep.csv")

############################################################ Data Processing

tidyData <- rawData
names(tidyData) <- c('satisfactionLevel','lastEvaluation','numberProject','averageMonthlyHours','tenure',
                     'accident','left','promotion','department','salary')
# Factorize salary and department
tidyData$salary <- factor(tidyData$salary,
                             levels = c('low','medium','high'))
tidyData$department <- factor(tidyData$department)

# Reorder columns so left(predictor variable) is first
tidyData <- tidyData %>%
  select(left,everything())

# Create training, validation and test sets
set.seed(6)
trainRows <- as.integer(sample(rownames(tidyData),dim(tidyData)[1]*.7))
validRows <- as.integer(sample(setdiff(rownames(tidyData),trainRows),
                    dim(tidyData)[1]*.2))
testRows <- as.integer(setdiff(rownames(tidyData),union(trainRows,validRows)))

trainData <- tidyData[trainRows,]
validData <- tidyData[validRows,]
testData <- tidyData[testRows,]

############################################################ EDA - on TrainData only

# Summary statistics
summaryStats <- data.frame(mean = sapply(trainData[2:8],mean),
                             median = sapply(trainData[2:8],median),
                             sd = sapply(trainData[2:8],sd),
                             min = sapply(trainData[2:8],min),
                             max = sapply(trainData[2:8],max))
knitr::kable(round(summaryStats,2), caption = 'Summary Statistics for Features')                           

# Count of Employees by Department (bar chart)
deptCount <- trainData %>%
  select(department) %>%
  group_by(department) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = reorder(department,-count), y=count))+
  geom_bar(stat = 'identity')+
  labs(x = 'Department', y = '', title = 'Count of Employees by Department')+
  geom_text(aes(label = count),vjust = 1.5, colour = 'white')+
  theme_classic()+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())+
  theme(axis.text.x = element_text(angle = 45,hjust = 1))
deptCount

# Salary distributions
salaryCount <- trainData %>%
  select(salary) %>%
  group_by(salary) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = salary, y=count))+
  geom_bar(stat = 'identity')+
  labs(x = 'Salary Group', y = '', title = 'Count of Employees by Salary Group')+
  geom_text(aes(label = count),vjust = 1.5, colour = 'white')+
  theme_classic()+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
salaryCount

# Visualize tenure (histogram)
tenurePlot <- trainData %>%
  select(tenure) %>%
  ggplot(aes(x = tenure))+
  geom_histogram(binwidth = 1, colour = 'black') +
  stat_bin(binwidth = 1,geom = 'text', aes(label = ..count..),vjust = 1, colour = 'white')+
  labs(x = 'Tenure', y= '',title = 'Tenure Distribution')+
  theme_classic()+
  scale_x_continuous(breaks = seq(0,10,by = 1))+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
tenurePlot  

# Heat map for correlations
#convert factor variables to numeric first so can calculate correlation
library(reshape)
covData <- trainData
covData$department <- as.numeric(covData$department)
covData$salary <- as.numeric(covData$salary)
cor.mat <- round(cor(covData),2)
melted.cor.mat <- melt(cor.mat)
heatMapCor <- melted.cor.mat %>%
  ggplot(aes(x = X1, y = X2, fill = value))+
  geom_tile()+
  geom_text(aes(x = X1, y = X2, label = value))+
  labs(x = '',y='', title = 'Correlation Matrix Between Variables')+
  theme_classic()+
  theme(axis.text.x = element_text(angle = 45,hjust = 1))
heatMapCor

############################################################ Modelling Employee Attrition
require(h2o)

#Convert training set to h2o format so the model can use it
h2o.init()
trainHrData <- as.h2o(trainData)
validHrData <- as.h2o(validData)
testHrData <- as.h2o(testData)

y='left'
x = setdiff(names(trainHrData),y)

set.seed(6)
h2oModel <- h2o.automl(x=x,
                       y =y,
                       training_frame = trainHrData,
                       validation_frame = validHrData,
                       max_runtime_secs = 50)

# Extract the most accurate model
autoBestModel <- h2oModel@leader

# Predict using best model on test data
bestModelPredict <- h2o.predict(object = autoBestModel, newdata = testHrData)

# Prep for performance assessment
test_performance <- testHrData %>%
  tibble::as_tibble() %>%
  select(left) %>%
  add_column(pred = as.vector(bestModelPredict$predict)) %>%
  mutate_if(is.character, as.factor)
test_performance$pred <- ifelse(test_performance$pred >.5,1,0)
h20ConfusionMatrix <- table(test_performance)

# Performance analysis
tn <- h20ConfusionMatrix[1]
tp <- h20ConfusionMatrix[4]
fp <- h20ConfusionMatrix[3]
fn <- h20ConfusionMatrix[2]

accuracy <- round((tp + tn) / (tp + tn + fp + fn),2)
misclassification_rate <- round(1 - accuracy,2)
true_positive_rate<- round(tp / (tp + fn),2)
precision <- round(tp / (tp + fp),2)
null_error_rate <- round(tn / (tp + tn + fp + fn),2)
true_negative_rate <- round(tn/(tn+fp),2)


h2oPerformance <- tibble(
  accuracy,
  misclassification_rate,
  true_positive_rate,
  precision,
  null_error_rate,
  true_negative_rate
) 

h2o.shutdown(prompt = FALSE) #close java virtual machine connection
#Precision - when left is 1 (leave), how often the model predicts 1
#Recall/specificy/true positive rate - when the left is 1, how often is the model correct? 