# Jurassic-Spark – CMU MLiP F25

Kafka Commands: 
```
ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf
lsof -ti:9092 | xargs kill -9
kcat -b  localhost:9092 -L 
kcat -b  localhost:9092 -t movielog8 -C -o -5 -c 5

```