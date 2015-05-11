using AdaGram
using NumericExtensions

vm, dict = load_model(ARGS[1])

while true
  println("Enter your question (empty line to exit): ")

  str = strip(readline())
  if length(str) == 0
    exit(0)
  end

  q = split(str)
  i = findfirst(q, "_")
  q = deleteat!(q, i)

  context =  Int32[get(dict.word2id, word, -1) for word in q]
  if length(context .== -1) > 0
    warn(string("Words not found in the dictionary: ", dict.id2word[context[context .== -1]]))
  end
  context = filter(w -> w > 0, context)

  const N = 4
  println("Enter $N answers: ")

  answers = Int32[]

  for j in 1:N
    ok = false
    while !ok
      word = lowercase(strip(readline()))
      w = get(dict.word2id, word, -1)
      ok = (w > 0)
      if !ok
        println("Word not found in the dictionary, try again")
      else
        push!(answers, w)
      end
    end
  end

  prob = zeros(N)
  for j in 1:N
    k = indmax(disambiguate(vm, answers[j], context))
    for y in context
      prob[j] += log_skip_gram(vm, answers[j], k, y)
    end
  end

  prob -= maximum(prob)
  exp!(prob)
  prob /= sum(prob)

  for j in 1:N
    println(dict.id2word[answers[j]], ": ", prob[j])
  end
end
