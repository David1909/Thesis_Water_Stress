
#set working directory
setwd("/Users/davidbokern/PycharmProjects/weigthed_mean_by_rev")

##A
file_name <- "A.txt"
A <- read.table(file_name, skip = 2, header =FALSE, sep ='\t')
A_cropped <- A
rm(A)
A_cropped[,2] <- NULL
A_cropped[,1] <- NULL
A <- A_cropped[-1,] #Now we have a 7987 * 7987 matrix (49 countries X  163 industries)
A <- data.matrix(A)
save(A, file="ixi/A.Rda") #re write this so the matrix is named A