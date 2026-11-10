;;;; Passage-based four-choice question generator for Stage 2 (single-hop reading comprehension and factual extraction).
;;;;
;;;; Every item is built from a small structured "world" (who lives where, who taught whom, who scored what, where someone went
;;;; in which order), rendered to a short passage with sentence frames, asked with a question frame, and checked by a solver that
;;;; recomputes the answer from the structured facts. Four options are always four entities of the asked type that ALL appear in
;;;; the passage, so "pick the option that is mentioned" is useless and the question has to be used.
;;;;
;;;; Leakage control is by construction:
;;;;   - entity pools (names, towns, jobs, pets, colours, instruments, places) are split 3/1/1 into train / dev / test with no overlap;
;;;;   - every sentence frame, question frame and passage intro exists for exactly one split (dev and test wording is written
;;;;     separately from training wording; bin/qa-build.lisp measures the remaining n-gram overlap);
;;;;   - answer positions are exactly balanced within every category of every split.
;;;; Same seed, same items (SBCL's seeded random state).
(defpackage :asem-qagen
  (:use :cl)
  (:export #:generate-split #:format-prompt #:item-prompt #:validate-item #:+letters+ #:+categories+
           #:item-json-line #:read-items #:all-frames #:frame-ngram-overlap #:split-pool #:*names* #:item-get))
(in-package :asem-qagen)

(defparameter +letters+ '("A" "B" "C" "D"))
(defparameter +categories+ '("extract" "reverse" "relation" "compare" "temporal"))

;;; ------------------------------------------------------------------ entity pools (split 3/1/1 by index: 0,1,2 train, 3 dev, 4 test)
(defparameter *names*
  '("Aaron" "Abel" "Ada" "Adam" "Adele" "Ahmed" "Aisha" "Alan" "Alba" "Alice" "Amir" "Amos" "Ana" "Andre" "Anita" "Anton" "Arjun"
    "Asha" "Aster" "Aziz" "Basil" "Bea" "Bella" "Ben" "Bianca" "Boris" "Bruno" "Carl" "Carla" "Cecil" "Celia" "Chen" "Clara" "Colin"
    "Cora" "Dalia" "Dan" "Dara" "Dev" "Diego" "Dina" "Dora" "Edgar" "Edith" "Elif" "Eli" "Ella" "Emil" "Emma" "Enzo" "Eric" "Erin"
    "Esra" "Eva" "Ezra" "Farah" "Felix" "Fern" "Finn" "Flora" "Frida" "Gabe" "Gail" "Gene" "Gia" "Gil" "Gina" "Greta" "Gus" "Hana"
    "Hank" "Hari" "Hazel" "Hugo" "Ian" "Ida" "Igor" "Ilse" "Ines" "Ira" "Iris" "Ivan" "Ivy" "Jack" "Jade" "Jan" "Jane" "Javi" "Jean"
    "Joel" "Jonas" "Jose" "Juan" "Kai" "Kara" "Kate" "Kemal" "Kim" "Kira" "Kofi" "Lars" "Lea" "Leo" "Lena" "Leon" "Lila" "Lina"
    "Liam" "Lou" "Luca" "Lucy" "Luis" "Luz" "Mae" "Mara" "Marco" "Maya" "Mei" "Mia" "Milo" "Mina" "Nadia" "Nate" "Neil" "Nell"
    "Nia" "Nico" "Nina" "Noah" "Nora" "Olga" "Omar" "Oren" "Otto" "Owen" "Paco" "Pam" "Paul" "Pedro" "Petra" "Pia" "Quinn" "Rafa"
    "Raj" "Ravi" "Rex" "Rhea" "Rita" "Rosa" "Ruby" "Rudy" "Ruth" "Sam" "Sara" "Saul" "Sean" "Selma" "Sid" "Sofia" "Sol" "Stan"
    "Tara" "Ted" "Tess" "Theo" "Tina" "Tom" "Uma" "Una" "Uri" "Vera" "Vic" "Vince" "Wade" "Wendy" "Xena" "Yara" "Yuki" "Yusuf"
    "Zack" "Zara" "Zeke" "Zoe"))

(defparameter *towns*
  (loop for on in '("Bar" "Tor" "Vel" "Kas" "Mir" "Dun" "Hal" "Fen" "Gor" "Lin" "Ost" "Pel" "Rav" "Sel" "Tam" "Ulm" "Var" "Wen" "Yor" "Zan")
        append (loop for su in '("vik" "ford" "by" "mere" "holm" "dale" "wick" "stad" "ton" "gard" "ley" "more")
                     collect (concatenate 'string on su))))

(defparameter *jobs*
  '("baker" "pilot" "nurse" "farmer" "tailor" "lawyer" "chemist" "plumber" "painter" "dentist" "sailor" "librarian" "carpenter"
    "florist" "banker" "chef" "miner" "poet" "jeweler" "surgeon" "architect" "butcher" "editor" "potter" "mechanic" "teacher" "judge"
    "weaver" "astronomer" "cashier" "dancer" "singer" "fisher" "gardener" "locksmith" "pharmacist" "photographer" "printer"
    "reporter" "translator" "vet" "welder"))

(defparameter *pets*
  '("cat" "dog" "parrot" "rabbit" "turtle" "goat" "hamster" "ferret" "pony" "goose" "lizard" "owl" "canary" "donkey" "pig" "sheep"
    "duck" "frog" "snake" "mouse" "rat" "horse" "llama" "hedgehog" "chicken" "tortoise" "gecko" "iguana" "cow" "alpaca"))

(defparameter *colors*
  '("red" "blue" "green" "yellow" "purple" "orange" "silver" "brown" "black" "white" "pink" "gray" "teal" "gold" "maroon" "navy"
    "beige" "violet" "crimson" "olive"))

(defparameter *instruments*
  '("violin" "drum" "flute" "piano" "guitar" "cello" "harp" "trumpet" "banjo" "clarinet" "oboe" "saxophone" "trombone" "tuba"
    "ukulele" "accordion" "harmonica" "bagpipes" "mandolin" "xylophone"))

(defparameter *places*
  '("bakery" "library" "market" "museum" "harbor" "post office" "gym" "bank" "clinic" "park" "station" "school" "garden" "cinema"
    "pharmacy" "barber" "zoo" "chapel" "theater" "castle" "farm" "factory" "stadium" "hotel" "cafe" "laundry" "airport" "beach"
    "forest" "pier"))

(defun split-pool (list split)
  (loop for x in list for i from 0
        when (ecase split (:train (< (mod i 5) 3)) (:dev (= (mod i 5) 3)) (:test (= (mod i 5) 4)))
          collect x))

;;; ------------------------------------------------------------------ frames. {P} person, {V} value, {aV} value with a/an,
;;; {N} number, {A} agent, {B} patient, {X} place. Each list belongs to exactly one split.
(defmacro by-split (&key train dev test) `(list :train ',train :dev ',dev :test ',test))

(defparameter *attr-types*
  (list
   (list :town :pool '*towns*
         :frames (by-split :train ("{P} lives in {V}." "{P} has lived in {V} for years." "{P}'s house is in {V}." "{V} is where {P} lives."
                                   "In {V} lives {P}." "{P} grew up in {V} and still lives there.")
                           :dev ("{P} resides in {V}." "{V} is the place where {P} resides.")
                           :test ("{P} makes a home in {V}." "The home of {P} stands in {V}." "{V} is home to {P}."))
         :q-extract (by-split :train ("Where does {P} live?" "Which town does {P} live in?" "In what town is {P}'s house?" "What town does {P} live in?")
                              :dev ("Where does {P} reside?" "What is {P}'s place of residence?")
                              :test ("Which town is home to {P}?" "Where is the home of {P}?"))
         :q-reverse (by-split :train ("Who lives in {V}?" "Which person lives in {V}?" "Whose house is in {V}?" "Who has a house in {V}?")
                              :dev ("Who resides in {V}?" "Whose place of residence is {V}?")
                              :test ("Who makes a home in {V}?" "{V} is home to which person?")))
   (list :job :pool '*jobs*
         :frames (by-split :train ("{P} works as {aV}." "{P} is {aV}." "{P} earns a living as {aV}." "By trade, {P} is {aV}." "{P} has a job as {aV}.")
                           :dev ("{P} is employed as {aV}." "{P}'s occupation is {V}.")
                           :test ("{P} spends each workday as {aV}." "The job of {P} is {V}." "{P} makes money working as {aV}."))
         :q-extract (by-split :train ("What does {P} do for work?" "What is {P}'s job?" "What job does {P} have?" "What does {P} work as?")
                              :dev ("What is {P}'s occupation?" "What is {P} employed as?")
                              :test ("Which line of work is {P} in?" "What kind of work does {P} do?"))
         :q-reverse (by-split :train ("Who works as {aV}?" "Who is {aV}?" "Which person has a job as {aV}?" "Who earns a living as {aV}?")
                              :dev ("Who is employed as {aV}?" "Whose occupation is {V}?")
                              :test ("Who makes money working as {aV}?" "Which person does the work of {aV}?")))
   (list :pet :pool '*pets*
         :frames (by-split :train ("{P} owns {aV}." "{P} has {aV} as a pet." "{P} takes care of {aV}." "{P}'s pet is {aV}." "{P} keeps {aV} at home.")
                           :dev ("{P} owns {aV} now." "The pet that {P} owns is {aV}.")
                           :test ("{P} lives with a pet {V}." "The pet of {P} is {aV}." "A pet {V} belongs to {P}."))
         :q-extract (by-split :train ("What pet does {P} own?" "What animal does {P} have?" "What is {P}'s pet?" "Which animal does {P} take care of?")
                              :dev ("What kind of pet does {P} own?" "Which animal does {P} own?")
                              :test ("Which pet animal belongs to {P}?" "Which animal is the pet of {P}?"))
         :q-reverse (by-split :train ("Who owns {aV}?" "Who has {aV} as a pet?" "Whose pet is {aV}?" "Who takes care of {aV}?")
                              :dev ("Which person owns {aV}?" "Who is the owner of {aV}?")
                              :test ("Who does the pet {V} belong to?" "Which person is the owner of the pet {V}?")))
   (list :color :pool '*colors*
         :frames (by-split :train ("{P}'s favorite color is {V}." "{P} likes the color {V} best." "The color {P} loves most is {V}."
                                   "{P} always picks {V} as a favorite color." "Of all colors, {P} prefers {V}.")
                           :dev ("{P} adores the color {V}." "{V} is the favorite color of {P}.")
                           :test ("{P} is fondest of the color {V}." "Nothing pleases {P} like the color {V}."))
         :q-extract (by-split :train ("What is {P}'s favorite color?" "Which color does {P} like best?" "What color does {P} prefer?" "Which color is {P}'s favorite?")
                              :dev ("Which color is the favorite of {P}?" "What color is {P} most fond of?")
                              :test ("Which color is {P} fondest of?" "Which color pleases {P} most?"))
         :q-reverse (by-split :train ("Whose favorite color is {V}?" "Who likes {V} best?" "Who prefers the color {V}?" "Which person loves {V} most?")
                              :dev ("Who is most fond of {V}?" "{V} is the favorite color of which person?")
                              :test ("Who is fondest of the color {V}?" "Which person is pleased most by {V}?")))
   (list :instrument :pool '*instruments*
         :frames (by-split :train ("{P} plays the {V}." "{P} practices the {V} every day." "{P} is learning to play the {V}."
                                   "The {V} is the instrument {P} plays." "{P} performs on the {V}.")
                           :dev ("{P} has mastered the {V}." "In the band, {P} is on the {V}.")
                           :test ("{P} gives concerts on the {V}." "The instrument of {P} is the {V}."))
         :q-extract (by-split :train ("What instrument does {P} play?" "Which instrument does {P} practice?" "What does {P} play?" "Which instrument is {P} learning?")
                              :dev ("Which instrument is associated with {P}?" "What is {P}'s instrument?")
                              :test ("What is the instrument of {P}?" "Which instrument does {P} make music with?"))
         :q-reverse (by-split :train ("Who plays the {V}?" "Who practices the {V}?" "Which person plays the {V}?" "Who is learning the {V}?")
                              :dev ("Who is associated with the {V}?" "Whose instrument is the {V}?")
                              :test ("Who makes music with the {V}?" "Which person is the {V} player?")))))

(defparameter *num-types*
  (list
   (list :age :range '(18 90)
         :frames (by-split :train ("{P} is {N} years old." "{P}'s age is {N}." "{P} just turned {N}." "{P} has reached the age of {N}.")
                           :dev ("{P} is aged {N}." "{P} was born {N} years ago.")
                           :test ("{P} has been alive for {N} years." "The age of {P} is {N}."))
         :q-extract (by-split :train ("How old is {P}?" "What is {P}'s age?" "What age is {P}?")
                              :dev ("How many years old is {P}?" "At what age is {P} now?")
                              :test ("For how many years has {P} been alive?" "What is the age of {P}?"))
         :q-max (by-split :train ("Who is the oldest?" "Which person is the oldest?" "Who is oldest of the four?")
                          :dev ("Who has the highest age?")
                          :test ("Who has been alive the longest?" "Of the four, who was born first?"))
         :q-min (by-split :train ("Who is the youngest?" "Which person is the youngest?" "Who is youngest of the four?")
                          :dev ("Who has the lowest age?")
                          :test ("Who has been alive the shortest time?" "Of the four, who was born last?")))
   (list :score :range '(3 99)
         :frames (by-split :train ("{P} scored {N} points." "{P} finished the game with {N} points." "{P} earned {N} points.")
                           :dev ("{P} got a total of {N} points.")
                           :test ("{P} collected {N} points in the contest." "In the contest, {N} points went to {P}."))
         :q-extract (by-split :train ("How many points did {P} score?" "How many points did {P} earn?")
                              :dev ("What total did {P} get?")
                              :test ("How many points did {P} collect in the contest?"))
         :q-max (by-split :train ("Who scored the most points?" "Who had the highest score?" "Which person earned the most points?")
                          :dev ("Who got the biggest total?")
                          :test ("Who won the contest by collecting the most points?" "Whose points total was the largest?"))
         :q-min (by-split :train ("Who scored the fewest points?" "Who had the lowest score?" "Which person earned the fewest points?")
                          :dev ("Who got the smallest total?")
                          :test ("Who collected the fewest points in the contest?" "Whose points total was the smallest?")))
   (list :height :range '(140 199)
         :frames (by-split :train ("{P} is {N} cm tall." "{P} stands {N} cm tall." "{P}'s height is {N} cm.")
                           :dev ("{P} measures {N} cm in height.")
                           :test ("{P} reaches {N} cm from floor to head." "The height of {P} is {N} cm."))
         :q-extract (by-split :train ("How tall is {P}?" "What is {P}'s height?")
                              :dev ("What does {P} measure in height?")
                              :test ("How many cm from floor to head is {P}?"))
         :q-max (by-split :train ("Who is the tallest?" "Which person is the tallest?")
                          :dev ("Who has the greatest height?")
                          :test ("Whose height is the greatest of all?" "Who would stand above all the others?"))
         :q-min (by-split :train ("Who is the shortest?" "Which person is the shortest?")
                          :dev ("Who has the least height?")
                          :test ("Whose height is the smallest of all?" "Who would look up at all the others?")))))

(defparameter *relations*
  (list
   (list :teach
         :frames (by-split :train ("{A} taught {B}." "{B} was taught by {A}." "{A} gave lessons to {B}.")
                           :dev ("{A} tutored {B}." "{B} studied under {A}.")
                           :test ("{A} was the teacher of {B}." "{B} learned from {A}."))
         :q-subj (by-split :train ("Who taught {B}?" "Who was {B} taught by?" "Who gave lessons to {B}?")
                           :dev ("Who tutored {B}?") :test ("Who was the teacher of {B}?" "From whom did {B} learn?"))
         :q-obj (by-split :train ("Who did {A} teach?" "Whom did {A} give lessons to?")
                          :dev ("Who did {A} tutor?") :test ("Who was a student of {A}?" "Who learned from {A}?")))
   (list :hire
         :frames (by-split :train ("{A} hired {B}." "{B} was hired by {A}." "{A} gave a job to {B}.")
                           :dev ("{A} recruited {B}.")
                           :test ("{B} got a job from {A}." "{A} took on {B} as a worker."))
         :q-subj (by-split :train ("Who hired {B}?" "Who gave a job to {B}?") :dev ("Who recruited {B}?")
                           :test ("From whom did {B} get a job?"))
         :q-obj (by-split :train ("Who did {A} hire?" "Who was hired by {A}?") :dev ("Who did {A} recruit?")
                          :test ("Who did {A} take on as a worker?")))
   (list :call
         :frames (by-split :train ("{A} called {B}." "{B} got a call from {A}." "{A} phoned {B}.")
                           :dev ("{A} rang {B}.")
                           :test ("{B} picked up a phone call from {A}." "{A} dialed the number of {B}."))
         :q-subj (by-split :train ("Who called {B}?" "Who phoned {B}?") :dev ("Who rang {B}?")
                           :test ("Whose phone call did {B} pick up?" "Who dialed the number of {B}?"))
         :q-obj (by-split :train ("Who did {A} call?" "Who got a call from {A}?") :dev ("Who did {A} ring?")
                          :test ("Whose number did {A} dial?")))
   (list :thank
         :frames (by-split :train ("{A} thanked {B}." "{B} was thanked by {A}.")
                           :dev ("{A} expressed gratitude to {B}.")
                           :test ("{B} received thanks from {A}." "{A} said thank you to {B}."))
         :q-subj (by-split :train ("Who thanked {B}?") :dev ("Who expressed gratitude to {B}?")
                           :test ("From whom did {B} receive thanks?"))
         :q-obj (by-split :train ("Who did {A} thank?" "Who was thanked by {A}?") :dev ("To whom did {A} express gratitude?")
                          :test ("Who did {A} say thank you to?")))
   (list :visit
         :frames (by-split :train ("{A} visited {B}." "{B} was visited by {A}." "{A} went to see {B}.")
                           :dev ("{A} paid a visit to {B}.")
                           :test ("{B} had {A} as a guest." "{A} dropped by the home of {B}."))
         :q-subj (by-split :train ("Who visited {B}?" "Who went to see {B}?") :dev ("Who paid a visit to {B}?")
                           :test ("Who was the guest of {B}?" "Who dropped by the home of {B}?"))
         :q-obj (by-split :train ("Who did {A} visit?" "Who was visited by {A}?") :dev ("Who did {A} pay a visit to?")
                          :test ("Whose home did {A} drop by?" "Who had {A} as a guest?")))
   (list :help
         :frames (by-split :train ("{A} helped {B}." "{B} was helped by {A}." "{A} gave help to {B}.")
                           :dev ("{A} assisted {B}.")
                           :test ("{B} got support from {A}." "{A} lent a hand to {B}."))
         :q-subj (by-split :train ("Who helped {B}?" "Who gave help to {B}?") :dev ("Who assisted {B}?")
                           :test ("Who lent a hand to {B}?" "From whom did {B} get support?"))
         :q-obj (by-split :train ("Who did {A} help?" "Who was helped by {A}?") :dev ("Who did {A} assist?")
                          :test ("Who got support from {A}?")))))

(defparameter *temporal*
  (list :first (by-split :train ("First, {P} went to the {X}." "{P} started the day at the {X}.")
                         :dev ("In the morning, {P} was at the {X}.")
                         :test ("{P}'s day began at the {X}."))
        :middle (by-split :train ("Then {P} went to the {X}." "Next, {P} stopped at the {X}." "After that, {P} visited the {X}.")
                          :dev ("Later on, {P} reached the {X}." "Once done there, {P} headed to the {X}.")
                          :test ("From there, {P} walked over to the {X}." "The next stop for {P} was the {X}."))
        :last (by-split :train ("Finally, {P} went to the {X}." "{P} ended the day at the {X}.")
                        :dev ("Last of all, {P} arrived at the {X}.")
                        :test ("{P}'s day ended at the {X}."))
        :q-first (by-split :train ("Where did {P} go first?" "What was the first place {P} went to?")
                           :dev ("Where did {P}'s day start?") :test ("Which was {P}'s first stop?"))
        :q-last (by-split :train ("Where did {P} go last?" "What was the last place {P} visited?")
                          :dev ("Where did {P}'s day finish?") :test ("At which place did {P}'s day end?"))
        :q-after (by-split :train ("Where did {P} go right after the {X}?" "Which place did {P} visit just after the {X}?")
                           :dev ("Where did {P} head once done at the {X}?") :test ("What was {P}'s next stop after the {X}?"))
        :q-before (by-split :train ("Where did {P} go right before the {X}?" "Which place did {P} visit just before the {X}?")
                            :dev ("Where was {P} prior to the {X}?") :test ("Which stop came immediately before the {X}?"))))

(defparameter *intros*
  (list :people (by-split :train ("" "" "Four neighbors share a street." "These facts are about four people." "Here is a note about four friends.")
                          :dev ("" "A record of four members follows.")
                          :test ("" "Below is a description of four coworkers." "The following concerns four classmates."))
        :relation (by-split :train ("" "" "Last week a few things happened in the office." "This is what happened at the club.")
                            :dev ("" "Events from the workshop are listed.")
                            :test ("" "Here is an account of the summer camp."))
        :temporal (by-split :train ("" "" "This is how {P} spent Saturday." "{P} had a busy day.")
                            :dev ("" "{P} ran errands on Monday.")
                            :test ("" "Here is a record of {P}'s errands."))))

(defun all-frames (split)
  "Every template string that belongs to SPLIT (for the overlap audit)."
  (let ((out nil))
    (flet ((grab (by) (dolist (s (getf by split)) (push s out))))
      (dolist (ty *attr-types*) (dolist (k '(:frames :q-extract :q-reverse)) (grab (getf (cdr ty) k))))
      (dolist (ty *num-types*) (dolist (k '(:frames :q-extract :q-max :q-min)) (grab (getf (cdr ty) k))))
      (dolist (r *relations*) (dolist (k '(:frames :q-subj :q-obj)) (grab (getf (cdr r) k))))
      (loop for (nil by) on *temporal* by #'cddr do (grab by))
      (loop for (nil by) on *intros* by #'cddr do (grab by)))
    (remove "" (nreverse out) :test #'string=)))

(defun frame-words (s)
  "Lower-case words of a frame with slot markers removed (a slot breaks the n-gram)."
  (let ((words nil) (cur (make-string-output-stream)) (n 0))
    (flet ((flush () (let ((w (get-output-stream-string cur))) (when (plusp (length w)) (push w words)))))
      (loop with in-slot = nil for c across (string-downcase s)
            do (cond ((char= c #\{) (flush) (push "|" words) (setf in-slot t))
                     ((char= c #\}) (setf in-slot nil))
                     (in-slot nil)
                     ((or (alphanumericp c) (char= c #\')) (write-char c cur) (incf n))
                     (t (flush))))
      (flush))
    (nreverse words)))

(defun frame-ngrams (s n)
  (let ((ws (frame-words s)))
    (loop for i from 0 to (- (length ws) n)
          for g = (subseq ws i (+ i n))
          unless (member "|" g :test #'string=) collect (format nil "~{~a~^ ~}" g))))

(defun frame-ngram-overlap (split-a split-b n)
  "Frames of SPLIT-A that share at least one slot-free word N-gram with a frame of SPLIT-B: list of (frame . shared-ngram)."
  (let ((seen (make-hash-table :test 'equal)) (out nil))
    (dolist (f (all-frames split-b)) (dolist (g (frame-ngrams f n)) (setf (gethash g seen) t)))
    (dolist (f (all-frames split-a) (nreverse out))
      (let ((hit (find-if (lambda (g) (gethash g seen)) (frame-ngrams f n))))
        (when hit (push (cons f hit) out))))))

;;; ------------------------------------------------------------------ helpers
(defun pick (list rng) (nth (random (length list) rng) list))
(defun shuffle (list rng)
  (let ((v (coerce list 'simple-vector)))
    (loop for i from (1- (length v)) downto 1 do (rotatef (svref v i) (svref v (random (1+ i) rng))))
    (coerce v 'list)))
(defun sample-n (list n rng) (subseq (shuffle list rng) 0 n))
(defun article (v) (if (find (char v 0) "aeiou") "an" "a"))
(defun upcase-first (s) (if (plusp (length s)) (concatenate 'string (string (char-upcase (char s 0))) (subseq s 1)) s))

(defun replace-all (text old new)
  (with-output-to-string (out)
    (let ((i 0) (n (length old)))
      (loop (let ((j (search old text :start2 i)))
              (cond (j (write-string text out :start i :end j) (write-string new out) (setf i (+ j n)))
                    (t (write-string text out :start i) (return))))))))

(defun fill-frame (frame &rest pairs)
  "PAIRS: :P name :V value :N number :A agent :B patient :X place. Sentence-initial values are capitalised."
  (let ((s frame))
    (loop for (k v) on pairs by #'cddr
          do (let ((v (if (numberp v) (princ-to-string v) v)))
               (when (eq k :v) (setf s (replace-all s "{aV}" (format nil "~a ~a" (article v) v))))
               (setf s (replace-all s (format nil "{~a}" (symbol-name k)) v))))
    (upcase-first s)))

(defun word-boundary-count (needle hay)
  "Occurrences of NEEDLE in HAY not glued to letters/digits on either side (case-insensitive)."
  (let ((n (length needle)) (count 0) (h (string-downcase hay)) (nd (string-downcase needle)))
    (loop with i = 0
          for j = (search nd h :start2 i)
          while j
          do (when (and (or (= j 0) (not (alphanumericp (char h (1- j)))))
                        (or (>= (+ j n) (length h)) (not (alphanumericp (char h (+ j n))))))
               (incf count))
             (setf i (1+ j)))
    count))

(defun sentences-of (passage)
  "Split on '. ' / '? ' boundaries (the frames never contain abbreviations)."
  (let ((out nil) (start 0))
    (loop for i from 0 below (length passage)
          do (when (and (find (char passage i) ".?!") (or (= i (1- (length passage))) (char= (char passage (1+ i)) #\Space)))
               (push (string-trim " " (subseq passage start (1+ i))) out) (setf start (1+ i))))
    (when (< start (length passage)) (push (string-trim " " (subseq passage start)) out))
    (remove "" (nreverse out) :test #'string=)))

(defun join-sentences (ss) (format nil "~{~a~^ ~}" (remove "" ss :test #'string=)))

;;; ------------------------------------------------------------------ worlds -> items
;;; Each builder returns a plist: :category :subtype :passage :question :answer-text :options-pool (four strings) :query
;;; :cf-question :cf-answer-text :qframe :facts. The caller places the answer at the requested option position.
(defun intro (kind split rng &rest pairs) (apply #'fill-frame (pick (getf (getf *intros* kind) split) rng) pairs))

(defun people-passage (split rng people type-a vals-a &optional type-b vals-b)
  "Sentences about each person; with a second attribute type the order is grouped by person or by type at random."
  (let* ((fa (getf (getf (cdr type-a) :frames) split))
         (sa (loop for p in people for v in vals-a collect (cons p (fill-frame (pick fa rng) :p p :v v :n v))))
         (sb (when type-b (let ((fb (getf (getf (cdr type-b) :frames) split)))
                            (loop for p in people for v in vals-b collect (cons p (fill-frame (pick fb rng) :p p :v v)))))))
    (cond ((null type-b) (mapcar #'cdr (shuffle sa rng)))
          ((zerop (random 2 rng)) (let ((order (shuffle people rng)))   ; grouped by person
                                    (loop for p in order append (shuffle (list (cdr (assoc p sa :test #'string=)) (cdr (assoc p sb :test #'string=))) rng))))
          (t (append (mapcar #'cdr (shuffle sa rng)) (mapcar #'cdr (shuffle sb rng)))))))

(defun distinct-numbers (lo hi n rng)
  (let ((out nil)) (loop while (< (length out) n) do (pushnew (+ lo (random (1+ (- hi lo)) rng)) out)) out))

(defun build-extract (split rng)
  (let* ((numeric (< (random 8 rng) 3))
         (type-a (if numeric (pick *num-types* rng) (pick *attr-types* rng)))
         (people (sample-n (split-pool *names* split) 4 rng))
         (vals (if numeric
                   (destructuring-bind (lo hi) (getf (cdr type-a) :range) (distinct-numbers lo hi 4 rng))
                   (sample-n (split-pool (symbol-value (getf (cdr type-a) :pool)) split) 4 rng)))
         (type-b (when (and (not numeric) (< (random 10 rng) 4)) (pick (remove type-a *attr-types*) rng)))
         (vals-b (when type-b (sample-n (split-pool (symbol-value (getf (cdr type-b) :pool)) split) 4 rng)))
         (sents (people-passage split rng people type-a vals type-b vals-b))
         (ti (random 4 rng)) (ci (mod (+ ti 1 (random 3 rng)) 4))
         (qf (pick (getf (getf (cdr type-a) :q-extract) split) rng)))
    (list :category "extract" :subtype (string-downcase (symbol-name (car type-a)))
          :passage (join-sentences (cons (intro :people split rng) sents))
          :question (fill-frame qf :p (nth ti people)) :qframe qf
          :answer-text (princ-to-string (nth ti vals)) :options-pool (mapcar #'princ-to-string vals)
          :cf-question (fill-frame qf :p (nth ci people)) :cf-answer-text (princ-to-string (nth ci vals))
          :query (list :extract (nth ti people)) :cf-query (list :extract (nth ci people))
          :facts (list :people people :values (mapcar #'princ-to-string vals)))))

(defun build-reverse (split rng)
  (let* ((type-a (pick *attr-types* rng))
         (people (sample-n (split-pool *names* split) 4 rng))
         (vals (sample-n (split-pool (symbol-value (getf (cdr type-a) :pool)) split) 4 rng))
         (type-b (when (< (random 10 rng) 4) (pick (remove type-a *attr-types*) rng)))
         (vals-b (when type-b (sample-n (split-pool (symbol-value (getf (cdr type-b) :pool)) split) 4 rng)))
         (sents (people-passage split rng people type-a vals type-b vals-b))
         (ti (random 4 rng)) (ci (mod (+ ti 1 (random 3 rng)) 4))
         (qf (pick (getf (getf (cdr type-a) :q-reverse) split) rng)))
    (list :category "reverse" :subtype (string-downcase (symbol-name (car type-a)))
          :passage (join-sentences (cons (intro :people split rng) sents))
          :question (fill-frame qf :v (nth ti vals)) :qframe qf
          :answer-text (nth ti people) :options-pool people
          :cf-question (fill-frame qf :v (nth ci vals)) :cf-answer-text (nth ci people)
          :query (list :reverse (nth ti vals)) :cf-query (list :reverse (nth ci vals))
          :facts (list :people people :values vals))))

(defun build-relation (split rng)
  (let* ((rel (pick *relations* rng))
         (people (sample-n (split-pool *names* split) 4 rng))
         (chain (zerop (random 2 rng)))
         (facts (if chain
                    (list (list (first people) (second people)) (list (second people) (third people)) (list (third people) (fourth people)))
                    (list (list (first people) (second people)) (list (third people) (fourth people)))))
         (frames (getf (getf (cdr rel) :frames) split))
         (sents (shuffle (mapcar (lambda (f) (fill-frame (pick frames rng) :a (first f) :b (second f))) facts) rng))
         (subj (zerop (random 2 rng)))
         (qf (pick (getf (getf (cdr rel) (if subj :q-subj :q-obj)) split) rng))
         (fi (random (length facts) rng)) (fc (mod (+ fi 1 (random (1- (length facts)) rng)) (length facts)))
         (ask (lambda (f) (if subj (fill-frame qf :b (second f)) (fill-frame qf :a (first f)))))
         (ans (lambda (f) (if subj (first f) (second f)))))
    (list :category "relation" :subtype (format nil "~(~a~)-~a-~a" (car rel) (if chain "chain" "pairs") (if subj "subj" "obj"))
          :passage (join-sentences (cons (intro :relation split rng) sents))
          :question (funcall ask (nth fi facts)) :qframe qf
          :answer-text (funcall ans (nth fi facts)) :options-pool people
          :cf-question (funcall ask (nth fc facts)) :cf-answer-text (funcall ans (nth fc facts))
          :query (if subj (list :rel-subj (second (nth fi facts))) (list :rel-obj (first (nth fi facts))))
          :cf-query (if subj (list :rel-subj (second (nth fc facts))) (list :rel-obj (first (nth fc facts))))
          :facts (list :pairs facts))))

(defun build-compare (split rng)
  (let* ((ty (pick *num-types* rng))
         (people (sample-n (split-pool *names* split) 4 rng))
         (vals (destructuring-bind (lo hi) (getf (cdr ty) :range) (distinct-numbers lo hi 4 rng)))
         (sents (people-passage split rng people ty vals))
         (mx (zerop (random 2 rng)))
         (qf (pick (getf (getf (cdr ty) (if mx :q-max :q-min)) split) rng))
         (cqf (pick (getf (getf (cdr ty) (if mx :q-min :q-max)) split) rng))
         (best (lambda (want-max) (nth (position (reduce (if want-max #'max #'min) vals) vals) people))))
    (list :category "compare" :subtype (format nil "~(~a~)-~a" (car ty) (if mx "max" "min"))
          :passage (join-sentences (cons (intro :people split rng) sents))
          :question (fill-frame qf) :qframe qf
          :answer-text (funcall best mx) :options-pool people
          :cf-question (fill-frame cqf) :cf-answer-text (funcall best (not mx))
          :query (list (if mx :max :min)) :cf-query (list (if mx :min :max))
          :facts (list :people people :numbers vals))))

(defun temporal-question (kind places split rng p)
  "(values question frame answer query) for KIND in :first :last :after :before."
  (let* ((qf (pick (getf (getf *temporal* (ecase kind (:first :q-first) (:last :q-last) (:after :q-after) (:before :q-before))) split) rng)))
    (ecase kind
      (:first (values (fill-frame qf :p p) qf (first places) (list :first)))
      (:last (values (fill-frame qf :p p) qf (fourth places) (list :last)))
      (:after (let ((i (random 3 rng))) (values (fill-frame qf :p p :x (nth i places)) qf (nth (1+ i) places) (list :after (nth i places)))))
      (:before (let ((i (1+ (random 3 rng)))) (values (fill-frame qf :p p :x (nth i places)) qf (nth (1- i) places) (list :before (nth i places))))))))

(defun build-temporal (split rng)
  (let* ((p (pick (split-pool *names* split) rng))
         (places (sample-n (split-pool *places* split) 4 rng))
         (sents (list (fill-frame (pick (getf (getf *temporal* :first) split) rng) :p p :x (first places))
                      (fill-frame (pick (getf (getf *temporal* :middle) split) rng) :p p :x (second places))
                      (fill-frame (pick (getf (getf *temporal* :middle) split) rng) :p p :x (third places))
                      (fill-frame (pick (getf (getf *temporal* :last) split) rng) :p p :x (fourth places))))
         (kind (pick '(:first :last :after :before) rng)))
    (multiple-value-bind (q qf a query) (temporal-question kind places split rng p)
      (multiple-value-bind (cq cqf ca cquery)
          (loop (multiple-value-bind (cq cqf ca cquery) (temporal-question (pick '(:first :last :after :before) rng) places split rng p)
                  (unless (string= ca a) (return (values cq cqf ca cquery)))))
        (declare (ignore cqf))
        (list :category "temporal" :subtype (string-downcase (symbol-name kind))
              :passage (join-sentences (cons (intro :temporal split rng :p p) sents))
              :question q :qframe qf :answer-text a :options-pool places
              :cf-question cq :cf-answer-text ca :query query :cf-query cquery
              :facts (list :person p :order places))))))

(defun builder (category)
  (cond ((string= category "extract") #'build-extract) ((string= category "reverse") #'build-reverse)
        ((string= category "relation") #'build-relation) ((string= category "compare") #'build-compare)
        ((string= category "temporal") #'build-temporal) (t (error "unknown category ~a" category))))

;;; ------------------------------------------------------------------ solver (recomputes answers from the structured facts)
(defun solve (category facts query)
  (let ((q (first query)) (arg (second query)))
    (cond ((string= category "extract")
           (nth (position arg (getf facts :people) :test #'string=) (getf facts :values)))
          ((string= category "reverse")
           (nth (position arg (getf facts :values) :test #'string=) (getf facts :people)))
          ((string= category "relation")
           (let ((hits (if (eq q :rel-subj)
                           (loop for (a b) in (getf facts :pairs) when (string= b arg) collect a)
                           (loop for (a b) in (getf facts :pairs) when (string= a arg) collect b))))
             (and (= 1 (length hits)) (first hits))))
          ((string= category "compare")
           (let* ((ns (getf facts :numbers)) (target (reduce (if (eq q :max) #'max #'min) ns)))
             (and (= 1 (count target ns)) (nth (position target ns) (getf facts :people)))))
          ((string= category "temporal")
           (let ((o (getf facts :order)))
             (ecase q (:first (first o)) (:last (car (last o)))
               (:after (let ((i (position arg o :test #'string=))) (and i (< i 3) (nth (1+ i) o))))
               (:before (let ((i (position arg o :test #'string=))) (and i (> i 0) (nth (1- i) o)))))))
          (t nil))))

;;; ------------------------------------------------------------------ items
(defun item-get (it k) (getf it k))

(defparameter *prompt-style* :compact
  "Prompt rendering. :COMPACT (used for all training and evaluation) is \"Text: ...\\nQ: ...\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nThe answer is\"
followed by the target (\" B\" in letter mode, \" <answer>\" in text mode). :ORIGINAL (\"Passage:\" / \"Question:\" / \"Answer:\") is 4 tokens longer;
it is kept only because data/qa/v1 was generated with its length check, so bin/qa-build.lisp reproduces v1 exactly.")

(defun format-prompt (passage question options &key (passage-p t) (question-p t) (style *prompt-style*))
  "The prompt for an item; see *PROMPT-STYLE*."
  (format nil (ecase style
                (:compact "~@[Text: ~a~%~]~@[Q: ~a~%~]A. ~a~%B. ~a~%C. ~a~%D. ~a~%The answer is")
                (:original "~@[Passage: ~a~%~]~@[Question: ~a~%~]A. ~a~%B. ~a~%C. ~a~%D. ~a~%Answer:"))
          (and passage-p passage) (and question-p question) (first options) (second options) (third options) (fourth options)))

(defun item-prompt (it &key (passage t) (question t) (question-text nil) (style *prompt-style*))
  (format-prompt (getf it :passage) (or question-text (getf it :question)) (getf it :options) :passage-p passage :question-p question :style style))

(defun validate-item (it)
  "List of problems (strings); NIL when the item is sound."
  (let* ((opts (getf it :options)) (ans (getf it :answer)) (passage (getf it :passage)) (cat (getf it :category)) (problems nil))
    (flet ((bad (fmt &rest args) (push (apply #'format nil fmt args) problems)))
      (unless (= 4 (length opts)) (bad "~d options" (length opts)))
      (unless (and (integerp ans) (<= 0 ans 3)) (bad "answer index ~a" ans))
      (unless (= 4 (length (remove-duplicates opts :test #'string-equal))) (bad "duplicate options"))
      (loop for a in opts do (loop for b in opts unless (eq a b) do (when (search (string-downcase a) (string-downcase b)) (bad "option ~s inside ~s" a b))))
      (dolist (o opts) (unless (plusp (word-boundary-count o passage)) (bad "option ~s not in the passage" o)))
      (when (and (integerp ans) (<= 0 ans 3) (getf it :facts))
        (let ((solved (solve cat (getf it :facts) (getf it :query))))
          (unless (and solved (string= solved (nth ans opts))) (bad "solver says ~s, key says ~s" solved (nth ans opts))))
        (let ((solved (solve cat (getf it :facts) (getf it :cf-query))))
          (unless (and solved (string= solved (nth (getf it :cf-answer) opts))) (bad "counterfactual solver ~s" solved))
          (when (eql (getf it :cf-answer) ans) (bad "counterfactual has the same answer"))))
      ;; text-level check, independent of the solver: for lookups, the sentences naming the asked entity contain exactly one
      ;; option, and it is the key
      (when (and (getf it :query) (member cat '("extract" "reverse") :test #'string=) (integerp ans) (<= 0 ans 3))
        (let* ((key (second (getf it :query)))
               (ss (remove-if-not (lambda (s) (plusp (word-boundary-count key s))) (sentences-of passage)))
               (found (remove-if-not (lambda (o) (some (lambda (s) (plusp (word-boundary-count o s))) ss)) opts)))
          (unless ss (bad "asked entity ~s not in the passage" key))
          (unless (equal found (list (nth ans opts))) (bad "sentences about ~s mention options ~s" key found))))
      (when (and (getf it :query) (string= cat "relation"))
        (unless (plusp (word-boundary-count (second (getf it :query)) passage)) (bad "asked person not in the passage")))
      (let ((counts (mapcar (lambda (o) (word-boundary-count o passage)) opts)))
        (when (and (member cat '("extract" "reverse" "compare" "temporal") :test #'string=) (notevery (lambda (c) (= c (first counts))) counts))
          (bad "options are mentioned unequally often ~a (a frequency shortcut)" counts))))
    (nreverse problems)))

(defun generate-split (split n seed &key (fits (constantly t)) (categories +categories+) (exclude (make-hash-table :test 'equal)))
  "N items for SPLIT (:train :dev :test), equal numbers per category, answer positions exactly balanced within each category
(N / (5 * 4) must be an integer for exact balance). FITS is called on each full prompt (and its ablations) and must return true
(the builder passes a token-length check). Prompts in EXCLUDE (normalised) are never produced; new ones are added to it.
Returns (values items rejected-count rejection-reasons)."
  (let* ((rng (sb-ext:seed-random-state seed)) (per (floor n (length categories))) (items nil) (rejected 0)
         (reasons (make-hash-table :test 'equal)))
    (loop for cat in categories
          do (dotimes (k per)
               (let ((pos (mod k 4)))
                 (loop
                   (let* ((w (funcall (builder cat) split rng))
                          (others (remove (getf w :answer-text) (shuffle (getf w :options-pool) rng) :test #'string=))
                          (opts (append (subseq others 0 pos) (list (getf w :answer-text)) (subseq others pos)))
                          (it (list :id (format nil "~(~a~)-~a-~4,'0d" split cat k) :split (string-downcase (symbol-name split))
                                    :category cat :subtype (getf w :subtype) :passage (getf w :passage) :question (getf w :question)
                                    :options opts :answer pos :cf-question (getf w :cf-question)
                                    :cf-answer (position (getf w :cf-answer-text) opts :test #'string=)
                                    :qframe (getf w :qframe) :query (getf w :query) :cf-query (getf w :cf-query) :facts (getf w :facts)))
                          (problems (validate-item it))
                          (key (string-downcase (item-prompt it :style :original))))
                     (cond (problems (incf rejected) (incf (gethash (first problems) reasons 0)))
                           ((gethash key exclude) (incf rejected) (incf (gethash "duplicate prompt" reasons 0)))
                           ((not (and (funcall fits (item-prompt it :style :original)) (funcall fits (item-prompt it :question-text (getf it :cf-question) :style :original))))
                            (incf rejected) (incf (gethash "too long for the context window" reasons 0)))
                           (t (setf (gethash key exclude) t) (push it items) (return))))))))
    (values (shuffle (nreverse items) rng) rejected
            (loop for k being the hash-keys of reasons using (hash-value v) collect (cons k v)))))

;;; ------------------------------------------------------------------ JSONL
(defun js (s) (asem-util:json-string s))

(defun item-json-line (it)
  (format nil "{\"id\":~a,\"split\":~a,\"category\":~a,\"subtype\":~a,\"passage\":~a,\"question\":~a,\"options\":[~{~a~^,~}],\"answer\":~d,\"answer_letter\":~a,\"cf_question\":~a,\"cf_answer\":~d,\"qframe\":~a,\"query\":~a,\"facts\":~a}"
          (js (getf it :id)) (js (getf it :split)) (js (getf it :category)) (js (getf it :subtype)) (js (getf it :passage))
          (js (getf it :question)) (mapcar #'js (getf it :options)) (getf it :answer) (js (nth (getf it :answer) +letters+))
          (js (getf it :cf-question)) (getf it :cf-answer) (js (getf it :qframe))
          (js (let ((*package* (find-package :keyword))) (prin1-to-string (list (getf it :query) (getf it :cf-query)))))
          (js (let ((*package* (find-package :keyword))) (prin1-to-string (getf it :facts))))))

(defun read-items (path)
  "Plists from a JSONL file written by ITEM-JSON-LINE."
  (with-open-file (in path :external-format :utf-8)
    (loop for line = (read-line in nil) while line
          when (plusp (length (string-trim " " line)))
            collect (let ((o (asem-json:parse-json line)))
                      (flet ((g (k) (asem-json:json-get o k)))
                        (list :id (g "id") :split (g "split") :category (g "category") :subtype (g "subtype") :passage (g "passage")
                              :question (g "question") :options (coerce (g "options") 'list) :answer (g "answer")
                              :cf-question (g "cf_question") :cf-answer (g "cf_answer") :qframe (g "qframe")))))))
