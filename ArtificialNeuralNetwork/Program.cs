using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Xml.Serialization;
using System.Threading;

namespace ArtificialNeuralNetwork
{
    class Program
    {
        const int cMaxIteration = 10000;
        Network _neo; // ссылка на нейросеть
        Pattern[] _patterns; // шаблоны обучения
        bool _flag = true;
        private List<Pattern> _patternList = new List<Pattern>();
        private Random randObj;
        private readonly int[] _layer = { 16, 132, 24 };//каждый элемент это слой, каждое значение количество нейронов 

        static void Main(string[] args)
        {
            Program main = new Program();
            main.OnStart();
        }

        private void OnStart()
        {
            randObj = new Random();
            _neo = LoadNet(_layer); //инициализация сети
            bool[] inter = new bool[16];//масив для удобства (быстрое переключение состояний)
            method(inter, inter.Length - 1);//осздаем  патерн обучения
            _patterns = new Pattern[_patternList.Count];
            _patterns = _patternList.ToArray();
            Console.WriteLine("Length = " + _patternList.Count);
            _neo.error = cheakNet(_patternList, _neo);
            Console.WriteLine("Error = " + _neo.error);

            do
            {
                //trainNet(_patterns); // тренеровка сети основанная на всех вариантах
                teachNet(_patterns,_neo); // альтернавивный вариант обучения
                //Console.WriteLine("Error = " + cheakNet(_patternList, _neo));
            } while (_neo.error != 0);

            //do
            //{
            //    teachNet(_patterns, _neo); // тренеровка сети основанная на всех вариантах
            //    SaveNet(_neo); //сохраним нейросеть
            //    Console.WriteLine("Error = " + cheakNet(_patternList, _neo));
            //} while (cheakNet(_patternList, _neo) > 0);

            //TestSpeedMethod();

            Console.WriteLine("\n\naccuracy = " + PatternNetAccuracy(_patternList, _neo) + " % ");
            Console.WriteLine("\nPress any key to exit.");
            System.Console.ReadKey();
        }

        private void TestSpeedMethod()
        {
            int max = 8000;
            int bs = 16;
            float[,] a = new float[max, max];
            float[,] b = new float[max, max];

            //init

            for (var i = 0; i < max; i++)
            {
                for (var j = 0; j < max; j++)
                {
                    a[i, j] = (float)j;
                    b[i, j] = (float)j;
                }
            }

            Stopwatch sWatch = new Stopwatch();
            sWatch.Start();

            //add(a, b, max);
            add(a, b, max, bs);

            sWatch.Stop();
            Console.WriteLine(sWatch.ElapsedMilliseconds.ToString());
        }

        //void add(float[,] a, float[,] b, int max)
        //{
        //    for (var i = 0; i < max; ++i)
        //    {
        //        for (var j = 0; j < max; ++j)
        //        {
        //            a[i, j] = a[i, j] + b[j, i];
        //        }
        //    }
        //}

        void add(float[,] a, float[,] b, int max, int bs)
        {
            for (var i = 0; i < max; i += bs)
            {
                for (var j = 0; j < max; j += bs)
                {
                    for (var ii = i; ii < i + bs; ii++)
                    {
                        for (var jj = j; jj < j + bs; jj++)
                        {
                            a[ii, jj] = a[ii, jj] + b[jj, ii];
                        }
                    }
                    //a[i, j] = a[i, j] + b[j, i];
                }
            }
        }

        //Пытаемся загрузить сохраненную сеть
        private Network LoadNet(int[] layer)
        {
            Network net;
            if (System.IO.File.Exists(@"D:\note.xml"))
            {
                XmlSerializer formatter = new XmlSerializer(typeof(Network));

                using (System.IO.FileStream fs = new System.IO.FileStream(@"D:\note.xml", System.IO.FileMode.OpenOrCreate))
                {
                    Network newNetwork = (Network)formatter.Deserialize(fs);
                    Console.WriteLine("Объект десериализован");
                    net = newNetwork;
                }
            }
            else
            {
                net = new Network(layer);
                Console.WriteLine("Объект не найден и создан новый");
            }
            return net;
        }

        //Пытаемся сохранить сеть
        private void SaveNet(Network net)
        {
            System.IO.File.Delete(@"D:\note.xml");
            XmlSerializer formatter = new XmlSerializer(typeof(Network));
            // получаем поток, куда будем записывать сериализованный объект
            using (System.IO.FileStream fs = new System.IO.FileStream(@"D:\note.xml", System.IO.FileMode.OpenOrCreate))
            {
                formatter.Serialize(fs, net);
                Console.Write(" Объект сериализован\n");
            }
        }

        //метод создания патернов Нужен boolean Массив для удобства
        private void method(bool[] bArray, int index)
        {
            for (int i = 0; i < 2; ++i)
            {
                bArray[index] = !bArray[index];
                if (index > 0)
                {
                    method(bArray, index - 1);
                }

                if (index == 0)
                {
                    int[] tempField = new int[bArray.Length]; //заготовка для шаблона
                    for (int j = bArray.Length - 1; j >= 0; j--)
                    {
                        tempField[j] = digit(!bArray[j]);
                    }

                    if (cheakField_4x4(tempField))
                    {
                        int[] answer = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4
                        opportunity_4x4(tempField, answer);
                        Pattern tempPattern = new Pattern(tempField, answer);
                        _patternList.Add(tempPattern);
                    }
                }//end if
            }//end for
        }//end method

        private int cheakNet(List<Pattern> patternList, Network net)
        {
            int error = 0;
            double[] inputs = new double[patternList[0].actualInputs.Length];// double 4x4
            double[] output = new double[patternList[0].actualOutput.Length];// выходные сигналы сигналы нейросети

            for (int a = 0; a < patternList.Count; ++a)
            {
                for (int i = 0; i < inputs.Length; ++i)// приведение входных сигналов к даблу
                {
                    inputs[i] = patternList[a].actualInputs[i];
                }
                net.RunNet(inputs);//получаем результат
                net.GetNetAnswer(output);// получаем выходные сигналы сигналы нейросети

                for (int i = 0; i < output.Length; ++i)
                {
                    double tempAccuracy = Math.Abs(patternList[a].actualOutput[i] - output[i]);
                    if (tempAccuracy > 0.5d)
                    {
                        error++;
                    }
                }
            }
            return error;
        }//end cheakNet

        private double PatternNetAccuracy(List<Pattern> patternList, Network net)
        {
            double accuracy = 0.0d;
            double tempAccuracy = 0.0d;
            double[] inputs = new double[patternList[0].actualInputs.Length];// double 4x4
            double[] output = new double[patternList[0].actualOutput.Length];// выходные сигналы сигналы нейросети

            for (int a = 0; a < patternList.Count; ++a)
            {
                tempAccuracy = 0.0d;
                for (int i = 0; i < inputs.Length; ++i)// приведение входных сигналов к даблу
                {
                    inputs[i] = patternList[a].actualInputs[i];
                }

                net.RunNet(inputs);//получаем результат 
                net.GetNetAnswer(output);// получаем выходные сигналы сигналы нейросети

                for (int i = 0; i < output.Length; ++i)
                {
                    tempAccuracy += Math.Abs(patternList[a].actualOutput[i] - output[i]);
                }
                tempAccuracy = 100.0d - ((tempAccuracy / output.Length) * 100);
                accuracy += tempAccuracy;
            }
            return (accuracy / patternList.Count);
        }

        private double NetAccuracy()
        {
            double accuracy = 0.0d;
            double tempAccuracy = 0.0d;
            int[] field = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4
            double[] inputs = new double[field.Length]; // double 4x4
            int[] answer = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4
            double[] output = new double[answer.Length]; ; // выходные сигналы сигналы нейросети
            int[] netAnswer = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4

            for (int a = 0; a < 100; ++a)
            {
                tempAccuracy = 0.0d;
                field = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4
                answer = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4

                for (int i = 0; i < field.Length; i++)
                {
                    field[i] = Rand(0, 1);
                }

                while (!cheakField_4x4(field))
                {
                    for (int i = 0; i < field.Length; i++)
                    {
                        field[i] = Rand(0, 1);
                    }
                }
                opportunity_4x4(field, answer);//заполняем ответы

                for (int i = 0; i < inputs.Length; ++i)// приведение входных сигналов к даблу
                {
                    inputs[i] = (double)field[i];
                }
                _neo.RunNet(inputs);//получаем результат 
                _neo.GetNetAnswer(output);// получаем выходные сигналы сигналы нейросети

                for (int i = 0; i < output.Length; ++i)
                {
                    tempAccuracy += Math.Abs((double)answer[i] - output[i]);
                }
                tempAccuracy = 100.0d - ((tempAccuracy / output.Length) * 100);
                accuracy += tempAccuracy;
            }
            return (double)(accuracy / 100.0d);
        }

        //метод создания из boolean цифру true==1, false==0
        private int digit(bool _bool)
        {
            if (_bool)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        private bool LearnNet()
        {
            int[] field = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4
            int[] answer = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };// 4x4

            for (int i = 0; i < field.Length; i++)
            {
                field[i] = Rand(0, 1);
                //Console.WriteLine(" " + field[i]);
            }

            if (!cheakField_4x4(field))
            {
                //LearnNet();
                return false;
            }

            _flag = !_flag;
            if (!opportunity_4x4(field, answer)) // сделано чтобы поля без ответа приходили не чаще чем 1 за 2 прохода
            {
                if (_flag)
                {
                    //LearnNet();
                    return false;
                }
                //Console.WriteLine("OFF");
            }
            Pattern pattern = new Pattern(field, answer);
            _neo.TrainNet(pattern.actualInputs, pattern.actualOutput); //обучаем сеть на основе патерна
            return true;
        }

        private bool opportunity_4x4(int[] field, int[] answer)
        {
            //Console.WriteLine("IN ");
            bool cheak = false;

            if (swap(field, 0, 4)) { cheak = true; answer[0] = 1; }
            if (swap(field, 1, 5)) { cheak = true; answer[1] = 1; }
            if (swap(field, 2, 6)) { cheak = true; answer[2] = 1; }
            if (swap(field, 3, 7)) { cheak = true; answer[3] = 1; }

            if (swap(field, 4, 8)) { cheak = true; answer[4] = 1; }
            if (swap(field, 5, 9)) { cheak = true; answer[5] = 1; }
            if (swap(field, 6, 10)) { cheak = true; answer[6] = 1; }
            if (swap(field, 7, 11)) { cheak = true; answer[7] = 1; }

            if (swap(field, 8, 12)) { cheak = true; answer[8] = 1; }
            if (swap(field, 9, 13)) { cheak = true; answer[9] = 1; }
            if (swap(field, 10, 14)) { cheak = true; answer[10] = 1; }
            if (swap(field, 11, 15)) { cheak = true; answer[11] = 1; }

            if (swap(field, 0, 1)) { cheak = true; answer[12] = 1; }
            if (swap(field, 1, 2)) { cheak = true; answer[13] = 1; }
            if (swap(field, 2, 3)) { cheak = true; answer[14] = 1; }

            if (swap(field, 4, 5)) { cheak = true; answer[15] = 1; }
            if (swap(field, 5, 6)) { cheak = true; answer[16] = 1; }
            if (swap(field, 6, 7)) { cheak = true; answer[17] = 1; }

            if (swap(field, 8, 9)) { cheak = true; answer[18] = 1; }
            if (swap(field, 9, 10)) { cheak = true; answer[19] = 1; }
            if (swap(field, 10, 11)) { cheak = true; answer[20] = 1; }

            if (swap(field, 12, 13)) { cheak = true; answer[21] = 1; }
            if (swap(field, 13, 14)) { cheak = true; answer[22] = 1; }
            if (swap(field, 14, 15)) { cheak = true; answer[23] = 1; }
            //Console.WriteLine("OUT ");
            return cheak;
        }

        private bool swap(int[] field, int first, int second)
        {
            bool cheak = false;
            int buf = 0;
            buf = field[first];
            field[first] = field[second];
            field[second] = buf;

            if (!cheakField_4x4(field)) cheak = true;

            buf = field[first];
            field[first] = field[second];
            field[second] = buf;

            return cheak;
        }

        //проверка на 3 в ряд (4x4)
        private bool cheakField_4x4(int[] field)
        {
            bool cheak = true;
            if (field[0] == 1 && field[1] == 1 && field[2] == 1) cheak = false;
            if (field[1] == 1 && field[2] == 1 && field[3] == 1) cheak = false;
            if (field[4] == 1 && field[5] == 1 && field[6] == 1) cheak = false;
            if (field[5] == 1 && field[6] == 1 && field[7] == 1) cheak = false;
            if (field[8] == 1 && field[9] == 1 && field[10] == 1) cheak = false;
            if (field[9] == 1 && field[10] == 1 && field[11] == 1) cheak = false;
            if (field[12] == 1 && field[13] == 1 && field[14] == 1) cheak = false;
            if (field[13] == 1 && field[14] == 1 && field[15] == 1) cheak = false;

            if (field[0] == 1 && field[4] == 1 && field[8] == 1) cheak = false;
            if (field[4] == 1 && field[8] == 1 && field[12] == 1) cheak = false;

            if (field[1] == 1 && field[5] == 1 && field[9] == 1) cheak = false;
            if (field[5] == 1 && field[9] == 1 && field[13] == 1) cheak = false;

            if (field[2] == 1 && field[6] == 1 && field[10] == 1) cheak = false;
            if (field[6] == 1 && field[10] == 1 && field[14] == 1) cheak = false;

            if (field[3] == 1 && field[7] == 1 && field[11] == 1) cheak = false;
            if (field[7] == 1 && field[11] == 1 && field[15] == 1) cheak = false;

            return cheak;
        }


        private int Rand(int minX, int maxX)
        {
            int finalX = RandomProvider.GetThreadRandom().Next(minX, maxX + 1);
            return finalX;
        }

        private void trainNet(Pattern[] patterns)
        {
            double[] inputs = new double[patterns[0].actualInputs.Length];// double 4x4
            double[] output = new double[patterns[0].actualOutput.Length];// выходные сигналы сигналы
            var count = 0;
            var max = 100;
            var errors = 0;
            for (int i = 0; i < max; i++)
            {
                System.Diagnostics.Stopwatch sw = new Stopwatch(); // time ---------------------------------
                sw.Start(); // time ------------------------------------------------------------------------

                for (int j = 0; j < patterns.Length; j++)
                {
                    _neo.TrainNet(patterns[j].actualInputs, patterns[j].actualOutput);
                }

                if (i % (max / 100) == 0)
                {
                    count++;
                    Console.SetCursorPosition(0, Console.CursorTop);
                    errors = cheakNet(_patternList, _neo);
                    Console.Write("Complete " + count + "% " + " errors = " + errors);
                    if (errors < _neo.error)
                    {
                        string str = String.Format("  dif = {0} ", _neo.error - errors);
                        _neo.error = errors;
                        Console.Write(str);
                        SaveNet(_neo); //сохраним нейросеть
                    }
                }

                sw.Stop();// time ----------------------------------------------------------------------------
                Console.ForegroundColor = ConsoleColor.DarkGreen; // устанавливаем цвет
                Console.WriteLine(" Iteration completed: {0}", (sw.ElapsedMilliseconds / 1000.0).ToString());
                Console.ResetColor();
            }
        }//end trainNet

        private double rand(double minX, double maxX)
        {
            double finalX = RandomProvider.GetThreadRandom().NextDouble() * (maxX - minX) + minX;
            return finalX;
        }


        /// <summary>
        /// Альтернативный вариант обучения сети (через ошибки)
        /// </summary>
        /// <param name="patterns">Патерны с вопросами и ответами</param>
        private void teachNet(Pattern[] patterns, Network net)
        {
            //double[] inputs = new double[patterns[0].actualInputs.Length];// double 4x4
            //double[] output = new double[patterns[0].actualOutput.Length];// выходные сигналы сигналы нейросети
            double[] inputs = new double[16];
            double[] output = new double[24];
            //net.Eta = 0.01d;

            System.Diagnostics.Stopwatch sw = new Stopwatch(); // time ---------------------------------
            sw.Start(); // time ------------------------------------------------------------------------

            for (var n = 0; n < patterns.Length; ++n)
            {
                net.RunNet(patterns[n].actualInputs);//получаем результат
                net.GetNetAnswer(output);// получаем выходные сигналы сигналы нейросети
                var error = 0;
                double tempAccuracy = 0;

                for (var i = 0; i < output.Length; ++i)
                {
                    tempAccuracy = Math.Abs(patterns[n].actualOutput[i] - output[i]);
                    //temp += tempAccuracy;
                    if (tempAccuracy > 0.50d)
                    {
                        ++error;
                    }
                }

                if (error != 0)
                {
                    net.TrainNet(patterns[n].actualInputs, patterns[n].actualOutput);
                    //--n;
                    //// Понижаем скорость обучения сети
                    //if (0.0001 > net.Eta)
                    //{
                    //    net.Eta = net.Eta - (net.Eta * 0.1);
                    //}
                    //continue;
                }
                //net.Eta = 0.01d;
                Console.SetCursorPosition(0, Console.CursorTop);
                //Console.Write("Complate " + (n * 100) / (patterns.Length) + " %     " + n + "          ");
            }

            sw.Stop();// time ----------------------------------------------------------------------------
            Console.ForegroundColor = ConsoleColor.DarkGreen; // устанавливаем цвет
            Console.WriteLine(" Iteration completed: {0}", (sw.ElapsedMilliseconds / 1000.0).ToString());
            Console.ResetColor();

            var errors = cheakNet(_patternList, net);
            if (errors < net.error)
            {
                string str = String.Format("  dif = {0}  current Error = {1} ", net.error - errors, errors);
                net.error = errors;
                Console.Write(str);
                SaveNet(net); //сохраним нейросеть
            }
        }
    }

}