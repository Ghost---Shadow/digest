# Microservices

## [10 Tips for failing badly at Microservices by David Schmitz](https://www.youtube.com/watch?v=X0tjziAQfNQ)

1. Go full scale Polyglot. 
* Polygot = knowing or using several languages.
* Not sticking to a fixed tech stack causes - context switching problems, redeveloping solutions for a different tech stack.
* Using in development libraries are a bad idea.
* Using multiple protocols is confusing. REST, HAL, SIREN
2. The data monolith
* Avoid microservices from sharing DB, if that is not possible, make all tables have single ownership.
* Connection Pools = In software engineering, a connection pool is a cache of database connections maintained so that the connections can be reused when future requests to the database are required. 
3. The event monolith
* WET = We enjoy typing. Opposite of DRY.
* If a message emmiter changes its format, all subscribers would have to change.
* Microservice version mismatch.
* Assume that other microservices can fail.
4. Think of the meat cloud
* WKPT
* hardcoding routes
6. The distributed monolith 
7. The SPA monolith
* Have a comprehensive requirement design.
* Each microservice can have one UI. Look up `tailor.js`. Hyperlinks to connect multiple FEs.
8. Decision monolith
* *Requirements ping pong*
* *Make developers create ppt*
* *Servant of many kings is a free man*
* Business Monolith
9. Use HR driven architecture
10. Staffing monolith

## [Microservices Anti-Patterns](https://www.youtube.com/watch?v=I56HzTKvZKc)

1. *No two services should talk to the DB at the same time*. Use a 3rd service. AKA gate keeper service. Add request version (semantic versioning) packet controling in this gate keeper service. 
2. Major version should be bumped when there is a backward incompatible change.
3. Queues smooth internal traffic. Redis works nicely as a capacitor.
4. Dont hardcode IPs and ports, use DNS with env vars. **Service Discovery Service**
5. Dogpile: Use circuit breakers

## [Moving Existing API From REST To GraphQL](https://www.youtube.com/watch?v=broQmxQAMjM)

1. REST is bad for depagination or repeated dynamic calls.
2. Single call instead of multiple for serverside filtering of data using aliases

Instead of 

```
GET /restaurant?name=mcdonalds&fetches=name,cuisine

GET /restaurant?name=wendys&fetches=name,cuisine
```

do this

```
{
  mcdonalds: restaurant(name: "mcdonalds") {
    name
    cuisine
  }
  wendys: restaurant(name: "wendys"){
    name
    cuisine
  }
}
```

## [Introduction to Apache Kafka by James Ward](https://www.youtube.com/watch?v=UEg40Te8pnE)

1. Guaranteed ordering, horizontal scaling
2. Kafka = distributed commit log, linear scaling
3. Records are sequential. This enables sequential replaying events. 
4. Partioning logic?
5. Delivery guarantes - Producer: Async(no gruarantees), Leader and Leader & quorum. Consumer: At least one, at most once, effectively once (dont process same request twice)
6. Log compaction: Keys get compacted. Array of objects to object of arrays?
7. Saves to disk, lets OS handle caching.
8. Pagecache to socket
9. auto rebalancing
10. Producer and consumer quotas (throttling)
11. Heroku kafka
12. Akka streams: consumer can throttle producer
13. Non text based sequelization is recommended

## [From CRUD to Event Sourcing Why CRUD is the wrong approach for microservices](https://www.youtube.com/watch?v=holjbuSbv3k)

1. Two generals problem: All acknowledgements need an acknowledgement. (unsolveable)
2. two phase commit: Reduce the window of failure by not committing to the commit in one go.
3. Event sourcing: Dont store state, store events. Events are deltas.

## [When Microservices Meet Event Sourcing](https://www.youtube.com/watch?v=cISNDnwlSgw)

1. Events are immutable and immortal.
2. Event store is not good for queries
3. Not good for read heavy apps.
