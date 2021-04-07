<?php
session_start();
if (isset($_SESSION['email']) && !empty($_SESSION['email'])) {
    $signed_in = true;
} else {
    $signed_in = false;
}
//
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $good_id = $_POST["good_id"];

    $servername = "localhost";
    $user = "root";
    $password = "";
    $dbname = "final_project";

// Create connection
    $conn = new mysqli($servername, $user, $password, $dbname);
    mysqli_set_charset($conn, "utf8");
// Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    if ($signed_in) {
        $person_id = 0;
        $sql = "SELECT id FROM person where email=\"" . $_SESSION['email'] . "\"";
        $result = $conn->query($sql);

        if ($result->num_rows > 0) {
            // output data of each row
            while ($row = $result->fetch_assoc()) {
                $person_id = $row["id"];
            }
        }
        //
        $sql = "INSERT INTO buying (good_id,person_id,status)
VALUES (" . $good_id . "," . $person_id . ",0)";
        $result = $conn->query($sql);
        header("location:../index.php");
    } else {
        echo 'برا اضافه کردن به سبد خرید ابتدا باید وارد حساب کاربری خود شوید.';
        header("location:../index.php");
    }
}

//


?>
