## [React Patterns in a Nutshell, with Kelly Andrews, Presented by Atlanta JavaScript](https://www.youtube.com/watch?v=C6w7R501oug)

1. Children Pass Throughs
2. Pass a function as the children, the parent calls the function with its own parameters

## [React Component Patterns by Michael Chan](https://www.youtube.com/watch?v=YaZg8wg39QQ)

1. higher order component: a component that returns a component

## [ReactNYC - Advanced Redux Patterns - Nir Kaufman (@nirkaufman)](https://www.youtube.com/watch?v=JUuic7mEs-s)

1. Action Types: Events, API calls, Transfer data to redux state
2. Middlewares - filtering, mapping, transform
3. Use a different reducer if testing?
4. One reducer can call multiple reducers. e.g. one should call API_REQUEST and other should update the UI
5. Transformation - enricher - e.g. append timestamp, normalizer - e.g. flatten structure. Translator - syntactic sugar?

## [Boris Dinkevich: Practical Advanced Redux — ReactNext 2017](https://www.youtube.com/watch?v=Gjiu7Lgdg3s)

1. Put throttling in middleware
2. Put analytics in middleware

## [10 tips for React Redux At Scale | Matters Meetup | Baptiste Manson](https://www.youtube.com/watch?v=NQta2urK3zk)

1. Redux ducks
2. Normalize states like DB tables

## [Effective React Testing - JazzCon 2018](https://www.youtube.com/watch?v=Eakp29J38YA)

1. testdouble.js
2. cypress.io
