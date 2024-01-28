library(data.table)
library(ggplot2)

#like <- fread("like.csv")
#like[, type := "like"]
unlike <- fread("unlike.csv")
unlike[, type := "unlike"]

#data <- rbind(like[, .(fhd, type)], unlike[, .(fhd, type)])
print(like)

ggplot(unlike, aes(x=fhd, fill=type)) + geom_histogram()
#ggplot(unlike, aes(x=fhd, fill=type)) + geom_density()
