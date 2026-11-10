;;;; Long-conversation memory benchmark generator.
;;;;
;;;; Each case is a synthetic chat: facts about the user are planted early, some are later
;;;; replaced by an update, third-party "distractor" statements reuse the same relation with a
;;;; different value, and questions come at the end. The generator keeps a symbolic ground-truth
;;;; store of (entity relation value) triples, so the expected answers are exact.
;;;;
;;;; Deterministic: the same seed gives the same output on any SBCL (own PRNG, no RANDOM).
(defpackage :asem-membench
  (:use :cl)
  (:export #:make-case #:case->json #:generate #:write-jsonl #:main
           #:make-rng #:rng-int #:*relations* #:*filler-topics*))
(in-package :asem-membench)

;;; ---------------------------------------------------------------- PRNG (splitmix64)

(defstruct (rng (:constructor %make-rng)) (state 0 :type (unsigned-byte 64)))
(defun make-rng (seed) (%make-rng :state (ldb (byte 64 0) seed)))

(defun rng-next (rng)
  (let* ((s (setf (rng-state rng) (ldb (byte 64 0) (+ (rng-state rng) #x9E3779B97F4A7C15))))
         (z (ldb (byte 64 0) (* (logxor s (ash s -30)) #xBF58476D1CE4E5B9)))
         (z (ldb (byte 64 0) (* (logxor z (ash z -27)) #x94D049BB133111EB))))
    (logxor z (ash z -31))))

(defun rng-int (rng n) (mod (rng-next rng) n))
(defun rng-float (rng) (/ (rng-int rng 1000000) 1000000d0))
(defun rng-pick (rng list) (nth (rng-int rng (length list)) list))

(defun shuffle (rng list)
  (let ((v (coerce list 'simple-vector)))
    (loop for i from (1- (length v)) downto 1
          do (rotatef (aref v i) (aref v (rng-int rng (1+ i)))))
    (coerce v 'list)))

;;; ---------------------------------------------------------------- world knowledge

;;; A relation: name, answers pool, templates for the user stating / updating the fact,
;;; one third-party template (~a person, ~a value), and the question.
(defstruct rel name values set update other question)

(defparameter *relations*
  (list
   (make-rel :name "lives_in"
             :values '("Paris" "Berlin" "Oslo" "Lisbon" "Denver" "Osaka" "Nairobi" "Lima" "Dublin" "Vienna" "Austin" "Seoul")
             :set '("I live in ~a." "We just settled in ~a, and the city is great." "These days my home is in ~a.")
             :update '("I moved to ~a last month." "Big news: I relocated to ~a." "I don't live where I used to; I'm in ~a now.")
             :other "~a lives in ~a."
             :question "Which city do I live in?")
   (make-rel :name "job"
             :values '("nurse" "carpenter" "pilot" "teacher" "accountant" "chef" "electrician" "translator" "architect" "paramedic" "baker" "surveyor")
             :set '("I work as a ~a." "I'm a ~a by trade." "My job is being a ~a.")
             :update '("I changed careers and I'm a ~a now." "After years of something else, I became a ~a." "I quit and retrained, so I work as a ~a these days.")
             :other "~a works as a ~a."
             :question "What is my profession?")
   (make-rel :name "dog"
             :values '("Biscuit" "Pepper" "Mango" "Ziggy" "Olive" "Waffles" "Juniper" "Rocket" "Maple" "Toby" "Clover" "Bruno")
             :set '("My dog is called ~a." "I have a dog named ~a." "Our family dog, ~a, keeps me busy.")
             :update '("We got a new dog and named him ~a." "Our old dog is gone; the new puppy is called ~a." "I adopted another dog, ~a, so he is my dog now.")
             :other "~a has a dog called ~a."
             :question "What is my dog's name?")
   (make-rel :name "food"
             :values '("ramen" "lasagna" "falafel" "tacos" "paella" "pierogi" "biryani" "dumplings" "risotto" "shakshuka" "pho" "curry")
             :set '("My favorite food is ~a." "I could eat ~a every day." "Nothing beats ~a for me.")
             :update '("I've changed my mind: my favorite food is ~a now." "I got tired of the old favorite; it's ~a these days." "Lately I only crave ~a.")
             :other "~a's favorite food is ~a."
             :question "What is my favorite food?")
   (make-rel :name "car"
             :values '("Corolla" "Civic" "Golf" "Outback" "Mustang" "Prius" "Fiesta" "Tucson" "Camry" "Focus" "Clio" "Panda")
             :set '("I drive a ~a." "My car is a ~a." "I own a ~a.")
             :update '("I sold my car and bought a ~a." "I upgraded, so I drive a ~a now." "My old car died; I own a ~a these days.")
             :other "~a drives a ~a."
             :question "What car do I drive?")
   (make-rel :name "hobby"
             :values '("chess" "pottery" "climbing" "birdwatching" "knitting" "sailing" "archery" "photography" "gardening" "fencing" "origami" "surfing")
             :set '("In my free time I do ~a." "My main hobby is ~a." "I'm really into ~a.")
             :update '("I dropped my old hobby and took up ~a." "My new hobby is ~a." "Lately all my free time goes into ~a.")
             :other "~a is really into ~a."
             :question "What is my hobby?")
   (make-rel :name "employer"
             :values '("Northwind" "Contoso" "Globex" "Initech" "Hooli" "Umbra" "Vandelay" "Stark Works" "Wayfinder" "Acme Freight" "Bluebird" "Tessellate")
             :set '("I work at ~a." "My employer is ~a." "I'm on the payroll at ~a.")
             :update '("I left and joined ~a." "I got a new employer: ~a." "My company changed; it's ~a now.")
             :other "~a works at ~a."
             :question "Which company do I work for?")
   (make-rel :name "instrument"
             :values '("cello" "trumpet" "banjo" "oboe" "accordion" "harp" "viola" "clarinet" "ukulele" "bassoon" "sitar" "flute")
             :set '("I play the ~a." "I've been learning the ~a for years." "My instrument is the ~a.")
             :update '("I switched instruments and now play the ~a." "I gave up the old one; it's the ~a for me now." "These days I practice the ~a.")
             :other "~a plays the ~a."
             :question "Which instrument do I play?")
   (make-rel :name "language"
             :values '("Italian" "Japanese" "Swahili" "Polish" "Greek" "Korean" "Dutch" "Turkish" "Finnish" "Hindi" "Czech" "Welsh")
             :set '("I'm learning ~a." "My current language study is ~a." "I practice ~a every morning.")
             :update '("I stopped that and started learning ~a instead." "I switched my studies to ~a." "Now I'm studying ~a.")
             :other "~a is learning ~a."
             :question "Which language am I learning?")
   (make-rel :name "sport"
             :values '("rugby" "squash" "volleyball" "rowing" "cricket" "handball" "judo" "cycling" "badminton" "hockey" "karate" "polo")
             :set '("I play ~a on weekends." "My sport is ~a." "I train for ~a twice a week.")
             :update '("I quit my sport and switched to ~a." "I play ~a now, not what I did before." "These days my sport is ~a.")
             :other "~a plays ~a."
             :question "Which sport do I play?")
   (make-rel :name "color"
             :values '("teal" "crimson" "amber" "indigo" "olive" "violet" "scarlet" "turquoise" "maroon" "lavender" "ochre" "navy")
             :set '("My favorite color is ~a." "I love the color ~a." "If I paint a room, it's ~a.")
             :update '("My favorite color changed to ~a." "I'm over my old favorite; now it's ~a." "Lately I prefer ~a.")
             :other "~a's favorite color is ~a."
             :question "What is my favorite color?")
   (make-rel :name "book"
             :values '("Dune" "Middlemarch" "Persuasion" "Neuromancer" "Ulysses" "Kokoro" "Emma" "Solaris" "Beloved" "Ficciones" "Kim" "Rebecca")
             :set '("My favorite novel is ~a." "I reread ~a every year." "The book I love most is ~a.")
             :update '("I finished a new favorite: ~a." "My favorite novel is ~a now." "~a replaced my old favorite book.")
             :other "~a's favorite novel is ~a."
             :question "What is my favorite novel?")))

(defparameter *people*
  '("my sister Maya" "my brother Tomas" "my friend Elena" "my coworker Raj" "my neighbor Ingrid"
    "my cousin Felix" "my roommate Aiko" "my uncle Boris" "my teacher Mrs. Okafor" "my mentor Daniel"))

(defparameter *ack*
  '("Thanks for telling me." "Good to know." "Noted." "That's interesting, thanks for sharing." "Okay, I'll keep that in mind."))

(defparameter *update-ack*
  '("Thanks for the update." "Understood, that's a change." "Got it, that's different from before." "Okay, noted."))

;;; Filler: a topic and a question about it, with a generic answer. None of it states a fact
;;; about the user or touches a relation above.
(defparameter *filler-topics*
  '(("how binary search works" "Binary search halves the sorted range each step until it finds the target.")
    ("why the sky looks blue" "Shorter wavelengths scatter more in the atmosphere, so blue dominates.")
    ("how to boil an egg" "Simmer it for about nine minutes, then cool it in cold water.")
    ("what a hash map is" "It maps keys to slots using a hash function for fast average lookups.")
    ("how tides work" "The moon's gravity pulls on the oceans, creating bulges that follow it around.")
    ("what compound interest means" "Interest is earned on the original sum and on interest already added.")
    ("how vaccines train the immune system" "They show the body a harmless piece of a germ so it can prepare a response.")
    ("how to reverse a linked list" "Walk the list while pointing each node back at its predecessor.")
    ("why leaves change color in autumn" "Chlorophyll breaks down and reveals yellow and orange pigments.")
    ("what a prime number is" "It is a natural number above one with no divisors except one and itself.")
    ("how airplanes stay up" "Wing shape and speed produce lift that exceeds the aircraft's weight.")
    ("how to write a good commit message" "Use a short summary line and explain the reason for the change.")
    ("what the water cycle is" "Water evaporates, condenses into clouds, and returns as precipitation.")
    ("how recursion works" "A function calls itself on a smaller input until it reaches a base case.")
    ("how to make sourdough starter" "Mix flour and water daily and let wild yeast build up over a week.")
    ("what an index does in a database" "It keeps a sorted structure so lookups avoid scanning every row.")
    ("why the seasons change" "Earth's axis is tilted, so sunlight hits each hemisphere differently over the year.")
    ("how a bill becomes a law" "It passes both chambers, then needs the executive's signature.")
    ("what photosynthesis is" "Plants turn light, water and carbon dioxide into sugar and oxygen.")
    ("how to stay focused while studying" "Work in short blocks, remove distractions and take real breaks.")
    ("how a compiler differs from an interpreter" "A compiler translates ahead of time, an interpreter executes as it reads.")
    ("what causes earthquakes" "Tectonic plates slip and release stored strain as seismic waves.")
    ("how to read a nutrition label" "Check the serving size first, then calories, sugar and sodium.")
    ("what a black hole is" "It is a region where gravity is so strong that not even light escapes.")
    ("how to fold a paper crane" "Start with a square, make the bird base, then shape the wings and head.")))

(defparameter *filler-openers*
  '("Can you explain ~a?" "Quick question: ~a?" "I was wondering about ~a." "Could you tell me about ~a?" "Remind me, ~a?"))

;;; ---------------------------------------------------------------- case structure

(defstruct (conv-case (:conc-name cc-))
  id seed exchanges turns questions facts)

;;; turn: plist (:role "user" :text "..." :event nil|(:kind :plant|:update|:other :relation "x" :value "y"))
;;; question: plist
;;; fact: plist describing the ground truth and its history

(defun fill-template (template &rest args) (apply #'format nil template args))

(defun filler-exchange (rng)
  (destructuring-bind (topic answer) (rng-pick rng *filler-topics*)
    (list (list :role "user" :text (fill-template (rng-pick rng *filler-openers*) topic))
          (list :role "assistant" :text answer))))

(defun string-capitalize-first (s)
  (if (zerop (length s)) s (concatenate 'string (string-upcase (subseq s 0 1)) (subseq s 1))))

(defun fact-exchange (rng kind rel value &optional person)
  "A user statement carrying an event of KIND (:plant :update :other) plus a short assistant ack."
  (let ((text (ecase kind
                (:plant (fill-template (rng-pick rng (rel-set rel)) value))
                (:update (fill-template (rng-pick rng (rel-update rel)) value))
                (:other (let ((s (fill-template (rel-other rel) (string-capitalize-first person) value)))
                          s)))))
    (list (list :role "user" :text text
                :event (list :kind kind :relation (rel-name rel) :value value :person person))
          (list :role "assistant" :text (rng-pick rng (if (eq kind :update) *update-ack* *ack*))))))

(defun pick-other (rng pool avoid)
  (let ((cands (remove-if (lambda (v) (member v avoid :test #'string=)) pool)))
    (rng-pick rng cands)))

(defun distinct-slots (rng lo hi k used)
  "K distinct exchange indices in [LO,HI) not in USED, in ascending order."
  (let ((cands (shuffle rng (loop for i from lo below hi unless (member i used) collect i))))
    (sort (subseq cands 0 (min k (length cands))) #'<)))

(defun make-case (id seed &key (exchanges 200) (facts 6) (update-fraction 0.5d0)
                              (distractor-rate 1.0d0) (absent-questions 2))
  "Build one case. EXCHANGES is the conversation length in user/assistant pairs. Facts are
planted in the first fifth, updates fall between 30% and 70%, distractors anywhere after the
first plant, and every question is asked after the last exchange."
  (let* ((rng (make-rng seed))
         (nrel (length *relations*))
         (facts (min facts (- nrel absent-questions)))
         (rels (shuffle rng *relations*))
         (planted (subseq rels 0 facts))
         (absent (subseq rels facts (+ facts absent-questions)))
         (early-hi (max (1+ facts) (floor exchanges 5)))
         (plant-at (distinct-slots rng 0 early-hi facts nil))
         (nupd (round (* facts update-fraction)))
         (updated (subseq (shuffle rng planted) 0 nupd))
         (upd-slots (distinct-slots rng (floor (* exchanges 3) 10) (floor (* exchanges 7) 10) nupd plant-at))
         (events (make-hash-table))         ; exchange index -> list of turns (2 elements)
         (facts-out nil)
         (used (append plant-at upd-slots))
         (ndis (round (* facts distractor-rate)))
         (dis-slots (distinct-slots rng (1+ (first plant-at)) exchanges ndis used)))
    (assert (>= exchanges (* 3 (+ facts absent-questions))) () "too few exchanges for the facts")
    ;; plant
    (loop for rel in (shuffle rng planted)
          for slot in plant-at
          do (let ((v (rng-pick rng (rel-values rel))))
               (setf (gethash slot events) (fact-exchange rng :plant rel v))
               (push (list :rel rel :value v :planted-at slot :current v :updated-at nil :stale nil :distractor nil) facts-out)))
    ;; update
    (loop for rel in updated
          for slot in (shuffle rng upd-slots)
          do (let* ((f (find rel facts-out :key (lambda (x) (getf x :rel))))
                    (nv (pick-other rng (rel-values rel) (list (getf f :value)))))
               (setf (gethash slot events) (fact-exchange rng :update rel nv))
               (setf (getf f :stale) (getf f :value)
                     (getf f :current) nv
                     (getf f :updated-at) slot)))
    ;; distractors: a third party with the same relation, value different from the user's values
    (loop for slot in dis-slots
          for f in (shuffle rng facts-out)
          do (let* ((rel (getf f :rel))
                    (avoid (remove nil (list (getf f :value) (getf f :current) (getf f :stale))))
                    (v (pick-other rng (rel-values rel) avoid))
                    (person (rng-pick rng *people*)))
               (setf (gethash slot events) (fact-exchange rng :other rel v person))
               (setf (getf f :distractor) (list :value v :person person :at slot))))
    ;; assemble turns
    (let ((turns nil))
      (dotimes (i exchanges)
        (dolist (tn (or (gethash i events) (filler-exchange rng)))
          (push tn turns)))
      (setf turns (nreverse turns))
      (let* ((qs (append
                  (loop for f in facts-out
                        collect (let* ((rel (getf f :rel)) (last (or (getf f :updated-at) (getf f :planted-at))))
                                  (list :relation (rel-name rel)
                                        :type (if (getf f :updated-at) "updated" "static")
                                        :question (rel-question rel)
                                        :expected (getf f :current)
                                        :stale (getf f :stale)
                                        :distractor (getf (getf f :distractor) :value)
                                        :planted-at (getf f :planted-at)
                                        :updated-at (getf f :updated-at)
                                        :distance (- exchanges last))))
                  (loop for rel in absent
                        collect (list :relation (rel-name rel) :type "absent"
                                      :question (rel-question rel) :expected nil :stale nil
                                      :distractor nil :planted-at nil :updated-at nil :distance nil))))
             (qs (shuffle rng qs)))
        (make-conv-case :id id :seed seed :exchanges exchanges :turns turns :questions qs
                        :facts (mapcar (lambda (f) (list :relation (rel-name (getf f :rel))
                                                         :entity "user"
                                                         :initial (getf f :value)
                                                         :current (getf f :current)))
                                       facts-out))))))

;;; ---------------------------------------------------------------- JSON output (one line per case)

(defun jstr (s) (asem-util:json-string s))
(defun jval (v) (cond ((null v) "null") ((stringp v) (jstr v)) ((integerp v) (format nil "~d" v)) (t (error "bad json value ~s" v))))

(defun case->json (c)
  (with-output-to-string (o)
    (format o "{\"id\":~a,\"seed\":~d,\"exchanges\":~d,\"words\":~d,\"turns\":["
            (jstr (cc-id c)) (cc-seed c) (cc-exchanges c)
            (loop for tn in (cc-turns c) sum (asem-util:count-words (getf tn :text))))
    (loop for (tn . rest) on (cc-turns c)
          do (format o "{\"role\":~a,\"text\":~a" (jstr (getf tn :role)) (jstr (getf tn :text)))
             (let ((ev (getf tn :event)))
               (when ev (format o ",\"event\":{\"kind\":~a,\"relation\":~a,\"value\":~a}"
                                (jstr (string-downcase (symbol-name (getf ev :kind))))
                                (jstr (getf ev :relation)) (jstr (getf ev :value)))))
             (write-string (if rest "}," "}") o))
    (write-string "],\"facts\":[" o)
    (loop for (f . rest) on (cc-facts c)
          do (format o "{\"entity\":\"user\",\"relation\":~a,\"initial\":~a,\"current\":~a}~a"
                     (jstr (getf f :relation)) (jstr (getf f :initial)) (jstr (getf f :current)) (if rest "," "")))
    (write-string "],\"questions\":[" o)
    (loop for (q . rest) on (cc-questions c)
          do (format o "{\"relation\":~a,\"type\":~a,\"question\":~a,\"expected\":~a,\"stale\":~a,\"distractor\":~a,\"planted_at\":~a,\"updated_at\":~a,\"distance\":~a}~a"
                     (jstr (getf q :relation)) (jstr (getf q :type)) (jstr (getf q :question))
                     (jval (getf q :expected)) (jval (getf q :stale)) (jval (getf q :distractor))
                     (jval (getf q :planted-at)) (jval (getf q :updated-at)) (jval (getf q :distance))
                     (if rest "," "")))
    (write-string "]}" o)))

(defun generate (n &key (base-seed 1) (exchanges 200) (facts 6) (update-fraction 0.5d0)
                     (distractor-rate 1.0d0) (absent-questions 2))
  (loop for i below n
        collect (make-case (format nil "c~4,'0d-n~d" i exchanges) (+ base-seed i)
                           :exchanges exchanges :facts facts :update-fraction update-fraction
                           :distractor-rate distractor-rate :absent-questions absent-questions)))

(defun write-jsonl (cases path)
  (with-open-file (s path :direction :output :if-exists :supersede :external-format :utf-8)
    (dolist (c cases) (write-string (case->json c) s) (terpri s)))
  path)

;;; ---------------------------------------------------------------- CLI

(defun parse-args (args)
  (let ((opts (list :out nil :cases 20 :lengths '(50 200 800) :seed 1 :facts 6 :updates 0.5d0)))
    (loop while args
          do (let ((a (pop args)))
               (cond ((string= a "--out") (setf (getf opts :out) (pop args)))
                     ((string= a "--cases") (setf (getf opts :cases) (parse-integer (pop args))))
                     ((string= a "--seed") (setf (getf opts :seed) (parse-integer (pop args))))
                     ((string= a "--facts") (setf (getf opts :facts) (parse-integer (pop args))))
                     ((string= a "--lengths")
                      (setf (getf opts :lengths) (mapcar #'parse-integer (asem-util:split-on (pop args) #\,))))
                     (t (error "unknown option ~a" a)))))
    opts))

(defun main (args)
  (let ((o (parse-args args)))
    (unless (getf o :out)
      (format t "usage: membench-gen.lisp --out FILE.jsonl [--cases N] [--lengths 50,200,800] [--facts K] [--seed S]~%")
      (sb-ext:exit :code 2))
    (let ((cases (loop for len in (getf o :lengths)
                       append (generate (getf o :cases) :base-seed (getf o :seed) :exchanges len
                                                        :facts (getf o :facts)
                                                        :update-fraction (getf o :updates)))))
      (write-jsonl cases (getf o :out))
      (format t "wrote ~d cases (~{~d~^,~} exchanges) to ~a~%" (length cases) (getf o :lengths) (getf o :out)))))
