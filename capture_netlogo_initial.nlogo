extensions [nw csv]

breed [guys guy]
breed [infobits infobit]

undirected-link-breed [friends friend]
undirected-link-breed [infolinks infolink]

globals [
  seed
  infobits-created
]

guys-own [
  group
  fluctuation
]

to setup
  clear-all
  set seed 42
  random-seed seed
  set infobits-created 0
  create-guys 20 [ initialize-guy ]
  ask guys [ set group random 2 ]

  ; Export initial positions
  let filename "netlogo_initial_positions.csv"
  csv:to-file filename [ (list [who] of guys [xcor] of guys [ycor] of guys [group] of guys) ]

  print "Initial positions exported to netlogo_initial_positions.csv"
end

to initialize-guy
  set shape "face happy"
  setxy random-xcor random-ycor
  set fluctuation 0
end
